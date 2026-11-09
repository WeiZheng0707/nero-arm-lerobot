"""Offline tests; no SDK hardware connection."""

import math
import unittest
from unittest.mock import Mock, patch

import test_pair_j1 as pair


class PairTests(unittest.TestCase):
    def test_accelerated_reference_velocity_and_acceleration(self):
        step = pair.ReferenceStep(math.radians(15), .02, math.radians(30))
        previous = velocity = 0.0
        for _ in range(400):
            current,_,_ = step.step(previous, math.radians(20), .02, (-1,1))
            new_velocity = (current-previous)/.02
            self.assertLessEqual(abs(new_velocity),math.radians(15)+1e-10)
            self.assertLessEqual(abs(new_velocity-velocity)/.02,math.radians(30)+1e-8)
            self.assertLessEqual(current,math.radians(20)+1e-10)
            previous,velocity=current,new_velocity
        self.assertAlmostEqual(previous,math.radians(20),places=7)

    def test_reversal_brakes_and_hard_limits_hold(self):
        step = pair.ReferenceStep(.2,.02,.4)
        previous=0
        for _ in range(50):
            previous,_,_=step.step(previous,.8,.02,(-.3,.3))
        old_velocity=step.velocity
        current,_,_=step.step(previous,-.3,.02,(-.3,.3))
        self.assertLessEqual(abs(step.velocity-old_velocity),.4*.02+1e-12)
        previous=current
        for _ in range(400):
            previous,_,_=step.step(previous,-.3,.02,(-.3,.3))
            self.assertTrue(-.3<=previous<=.3)
        self.assertAlmostEqual(previous,-.3,places=7)

    def test_high_rate_does_not_apply_old_50ms_step(self):
        mapper=pair.SevenJointMapper([0]*7,[0]*7,speed_deg_s=10,period=.02)
        result=mapper.target([1]*7,10)
        self.assertAlmostEqual(result[0],math.radians(.2))

    def test_motion_rates_use_real_elapsed_time(self):
        rates=pair.MotionRates([0]*7,[0]*7,[0]*7,0)
        result=rates.update([math.radians(2)]*7,[math.radians(1)]*7,[math.radians(.5)]*7,.1)
        self.assertAlmostEqual(result[0][0],20)
        self.assertAlmostEqual(result[1][0],10)
        self.assertAlmostEqual(result[2][0],5)

    def test_invalid_control_parameters(self):
        for fps,percent,accel in ((19,5,None),(61,5,None),(50,31,20),(50,0,20),
                                  (50,15,0),(50,15,float('nan')),(50,15,61)):
            with self.assertRaises(ValueError):
                pair.validate_control(fps,percent,accel)

    def test_continuous_interrupt_cleans_up_both_devices(self):
        now=[0.0]
        l,lg,f,fg=[Mock() for _ in range(4)]
        limits={f'joint{i}':[-2,2] for i in range(1,8)}
        f.get_config.return_value={'joint_limits':limits}
        f.move_j.side_effect=KeyboardInterrupt
        leader_frames=[None,(1,[0]*7),(2,[0]*7),(3,[.2]*7)]
        with patch('builtins.input',return_value='START'), \
                patch.object(pair.time,'monotonic',side_effect=lambda:now[0]), \
                patch.object(pair.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)), \
                patch.object(pair,'wait_state',return_value=([1]*7,[False]*7,[])), \
                patch.object(pair,'stable_pose',return_value=[0]*7), \
                patch.object(pair,'enable_checked',return_value=Mock()), \
                patch.object(pair,'status_snapshot',return_value=(1,100,0,0,1,'J')), \
                patch.object(pair,'StatusMonitor',return_value=Mock()), \
                patch.object(pair,'leader_pose',side_effect=leader_frames), \
                patch.object(pair,'pose',side_effect=[(1,[0]*7),(2,[0]*7)]), \
                patch.object(pair,'snapshot',return_value=([1]*7,[True]*7,[])), \
                patch.object(pair,'cleanup') as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                pair.session(l,lg,f,fg,None,all_joints=True,full_range=True,speed_deg_s=15,
                             fps=50,follower_speed_percent=20,accel_deg_s2=30)
            cleanup.assert_called_once_with({'follower':(f,fg),'leader':(l,lg)},True)
            f.move_j.assert_called_once()
            self.assertEqual(f.set_speed_percent.call_args.args,(20,))

    def test_full_range_removes_test_envelope_but_keeps_sdk_bounds_and_speed(self):
        limits={f'joint{i}':[-.4,.6] for i in range(1,8)}
        mapper=pair.SevenJointMapper([0]*7,[.1]*7,speed_deg_s=5,full_range=True,limits=limits)
        previous=[.1]*7
        for _ in range(250):
            target=mapper.target([2,-2,.25,0,0,0,0],10)
            for i in range(7):
                self.assertLessEqual(abs(target[i]-previous[i]),math.radians(.25)+1e-12)
                self.assertTrue(-.4-1e-12<=target[i]<=.6+1e-12)
            previous=target
        self.assertAlmostEqual(target[0],.6)
        self.assertAlmostEqual(target[1],-.4)
        self.assertAlmostEqual(target[2],.35)
        self.assertEqual(mapper.range_limited,[])
        self.assertEqual(mapper.joint_limited,[1,2])

    def test_full_range_requires_limits(self):
        with self.assertRaises(ValueError):
            pair.SevenJointMapper([0]*7,[0]*7,full_range=True)

    def test_continuous_cancel_has_no_hardware_changes(self):
        devices=[Mock() for _ in range(4)]
        with patch.object(pair,'wait_state',return_value=([1]*7,[False]*7,[])), \
                patch('builtins.input',return_value='NO'):
            pair.session(*devices,None,all_joints=True,full_range=True)
        devices[0].enable.assert_not_called()
        devices[2].enable.assert_not_called()

    def test_tuned_mapping_bounds_and_limit_indicators(self):
        mapper = pair.SevenJointMapper([0]*7,[0]*7,5,2)
        target = mapper.target([math.radians(10),math.radians(-10),0,0,0,0,0],10)
        self.assertAlmostEqual(target[0],math.radians(.1))
        self.assertAlmostEqual(target[1],-math.radians(.1))
        self.assertEqual(mapper.range_limited,[1,2])
        self.assertEqual(mapper.speed_limited,[1,2])
        for _ in range(100):
            target = mapper.target([math.radians(10),math.radians(-10),0,0,0,0,0],.05)
        self.assertAlmostEqual(target[0],math.radians(5))
        self.assertEqual(mapper.speed_limited,[])
        mapper.target([math.radians(5),math.radians(-5),0,0,0,0,0],.05)
        self.assertEqual(mapper.range_limited,[])

    def test_tuned_j1_keeps_other_axes_fixed(self):
        mapper=pair.J1Mapper(0,[0.1]*7,5,2)
        target=mapper.target(math.radians(10),.05)
        self.assertAlmostEqual(target[0]-.1,math.radians(.1))
        self.assertEqual(target[1:],[.1]*6)
        self.assertEqual(mapper.range_limited,[1])
        self.assertEqual(mapper.speed_limited,[1])

    def test_invalid_tuning_rejected(self):
        for range_deg,speed in ((0,1),(16,1),(3,0),(3,31),(float('nan'),1),(3,float('inf'))):
            with self.assertRaises(ValueError):
                pair.validate_tuning(range_deg,speed)

    def test_shared_window_recovers_from_initial_motion(self):
        now = [0.0]
        def read(arm):
            value = math.radians(1) if arm == 'f' and now[0] >= .4 else 0
            return 1+now[0], [value]*7
        with patch.object(pair.time,'monotonic',side_effect=lambda:now[0]), \
                patch.object(pair.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)), \
                patch.object(pair,'pose',side_effect=read):
            result = pair.stable_poses({'leader':'l','follower':'f'},timeout=4)
        self.assertGreaterEqual(now[0],1.4)
        self.assertEqual(result['follower'],[math.radians(1)]*7)

    def test_continuous_drift_and_stale_frames_rejected(self):
        for stale in (True,False):
            now = [0.0]
            def read(arm):
                return (1 if stale else 1+now[0]), [math.radians(now[0])]*7
            with patch.object(pair.time,'monotonic',side_effect=lambda:now[0]), \
                    patch.object(pair.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)), \
                    patch.object(pair,'pose',side_effect=read):
                with self.assertRaisesRegex(RuntimeError,'No stable pose captured'):
                    pair.stable_poses({'follower':Mock()},timeout=2)

    def test_missing_saved_file_stops_before_sdk_import(self):
        with patch('sys.argv',['test_pair_j1.py','--leader-channel','0','--follower-channel','1',
                               '--return-to-start']), \
                patch.object(pair.presets,'load',side_effect=FileNotFoundError):
            with self.assertRaises(SystemExit) as exc:
                pair.main()
            self.assertEqual(exc.exception.code,2)

    def test_seven_axes_identity_at_start_and_bounds(self):
        anchor = [1.0]*7
        start = [0.1]*7
        mapper = pair.SevenJointMapper(anchor,start)
        self.assertEqual(mapper.target(anchor,.05),start)
        previous = list(start)
        for _ in range(300):
            target = mapper.target([1+(-1)**i for i in range(7)], 5)
            for i in range(7):
                self.assertLessEqual(abs(target[i]-start[i]),math.radians(3)+1e-12)
                self.assertLessEqual(abs(target[i]-previous[i]),math.radians(.025)+1e-12)
            previous=target
        for i in range(7):
            self.assertAlmostEqual(target[i]-start[i],(-1)**i*math.radians(3))

    def test_seven_axes_reject_invalid_feedback(self):
        for angles in ([0]*6,[float('nan')]*7):
            with self.assertRaises(ValueError):
                pair.SevenJointMapper([0]*7,[0]*7).target(angles,.05)

    def test_relative_start_is_stationary(self):
        mapper = pair.J1Mapper(1.5, [0.1]*7)
        self.assertEqual(mapper.target(1.5, .05), [.1]*7)

    def test_bounded_both_directions_and_other_joints_fixed(self):
        for direction in (-1, 1):
            start = [.1, .2, .3, .4, .5, .6, .7]
            mapper = pair.J1Mapper(1.5, start)
            previous = start
            for _ in range(300):
                value = mapper.target(1.5 + direction, .05)
                self.assertEqual(value[1:], start[1:])
                self.assertLessEqual(abs(value[0]-start[0]), math.radians(3)+1e-12)
                self.assertLessEqual(abs(value[0]-previous[0]), math.radians(.025)+1e-12)
                previous = value
            self.assertAlmostEqual(value[0]-start[0], direction*math.radians(3))

    def test_delayed_tick_does_not_jump(self):
        mapper = pair.J1Mapper(0,[0]*7)
        self.assertAlmostEqual(mapper.target(1,10)[0], math.radians(.025))

    def test_invalid_values_rejected(self):
        for angle,dt in ((float('nan'),.05),(float('inf'),.05),(1,-1),(1,float('nan'))):
            with self.assertRaises(ValueError):
                pair.J1Mapper(0,[0]*7).target(angle,dt)

    def test_cleanup_attempts_both_sides_on_failure(self):
        l,lg,f,fg = [Mock() for _ in range(4)]
        with patch.object(pair,'finish',side_effect=[RuntimeError('follower fail'),None]) as finish:
            with self.assertRaisesRegex(RuntimeError,'UNCONFIRMED'):
                pair.cleanup({'leader':(l,lg),'follower':(f,fg)},True)
            self.assertEqual(finish.call_count,2)
            self.assertEqual(finish.call_args_list[0].args,(f,fg,False))
            self.assertEqual(finish.call_args_list[1].args,(l,lg,True))

    def test_cancel_never_changes_mode_or_enables(self):
        devices = [Mock() for _ in range(4)]
        with patch.object(pair,'wait_state',return_value=([1]*7,[False]*7,[])), \
                patch('builtins.input',return_value='NO'):
            pair.session(*devices,5)
        for arm in (devices[0],devices[2]):
            arm.enable.assert_not_called()
            arm.move_j.assert_not_called()
            arm.set_leader_mode.assert_not_called()

    def test_follower_prepare_failure_is_cleaned_up(self):
        devices = [Mock() for _ in range(4)]
        devices[2].get_config.return_value = {'joint_limits': {}}
        with patch.object(pair,'wait_state',return_value=([1]*7,[False]*7,[])), \
                patch('builtins.input',return_value='START'), \
                patch.object(pair,'stable_pose',return_value=[0]*7), \
                patch.object(pair,'check_limits'), \
                patch.object(pair,'enable_checked',side_effect=RuntimeError('enable failure')), \
                patch.object(pair,'cleanup') as cleanup:
            with self.assertRaisesRegex(RuntimeError,'enable failure'):
                pair.session(*devices,5)
            cleanup.assert_called_once_with({'follower':(devices[2],devices[3])},False)
        devices[0].enable.assert_not_called()


if __name__ == '__main__':
    unittest.main()
