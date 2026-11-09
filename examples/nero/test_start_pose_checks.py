"""Offline saved-pose tests; never connects to hardware."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import start_pose as presets
import test_pair_j1 as pair


class StartPoseTests(unittest.TestCase):
    def test_roundtrip_and_no_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"pose.json"
            presets.save(path,[.1]*7,[.2]*7)
            data = presets.load(path)
            self.assertEqual(data['leader']['joints'],[.1]*7)
            self.assertEqual(data['unit'],'rad')
            with self.assertRaises(FileExistsError):
                presets.save(path,[0]*7,[0]*7)
            presets.save(path,[0]*7,[0]*7,overwrite=True)
            self.assertEqual(presets.load(path)['follower']['joints'],[0]*7)

    def test_invalid_pose_rejected_before_write(self):
        for invalid in ([0]*6,[float('nan')]*7,[True]*7):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"pose.json"
                with self.assertRaises(ValueError):
                    presets.save(path,invalid,[0]*7)
                self.assertFalse(path.exists())

    def test_unit_and_firmware_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            presets.validate({'schema':1,'unit':'deg'})
        with self.assertRaises(ValueError):
            presets.validate({'schema':1,'unit':'rad',
                              'leader':{'firmware':'1.21','joints':[0]*7}})

    def test_reference_step_bound_and_no_overshoot(self):
        value = presets.next_target([0]*7,[1,-1,0,.0001,0,0,0],100)
        self.assertAlmostEqual(value[0],math.radians(.05))
        self.assertAlmostEqual(value[1],-math.radians(.05))
        self.assertEqual(value[3],.0001)

    def test_faster_return_step_stays_bounded(self):
        value=presets.next_target([0]*7,[1,-1,0,.0001,0,0,0],10,speed_deg_s=2)
        self.assertAlmostEqual(value[0],math.radians(.1))
        self.assertAlmostEqual(value[1],-math.radians(.1))
        self.assertEqual(value[3],.0001)

    def test_invalid_return_parameters(self):
        for speed,percent in ((0,5),(6,5),(float('nan'),5),(2,0),(2,21),(2,True)):
            with self.assertRaises(ValueError):
                presets.validate_return_tuning(speed,percent)
        with self.assertRaises(ValueError):
            presets.next_target([0]*7,[1]*7,float('inf'))

    def simulate(self, frozen=False, follows=True, goal=None, **settings):
        now=[0.0]
        actual=[[0.0]*7]
        arm=Mock()
        arm.get_config.return_value={'joint_limits':{f'joint{i}':[-2,2] for i in range(1,8)}}
        if follows:
            arm.move_j.side_effect=lambda target: actual.__setitem__(0,list(target))
        with patch.object(presets.time,'monotonic',side_effect=lambda:now[0]), \
                patch.object(presets.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)), \
                patch.object(presets,'pose',side_effect=lambda a:(1 if frozen else 1+now[0],list(actual[0]))), \
                patch.object(presets,'snapshot',return_value=([1]*7,[True]*7,[])):
            presets.return_to_pose(arm,goal or [math.radians(1)]*7,Mock(),'follower',**settings)
        return arm,actual[0]

    def test_return_success_without_enable_or_gripper_commands(self):
        arm,actual=self.simulate()
        self.assertAlmostEqual(actual[0],math.radians(1))
        arm.enable.assert_not_called()
        arm.init_effector.assert_not_called()

    def test_faster_return_controller_and_step(self):
        arm,actual=self.simulate(speed_deg_s=2,controller_percent=10)
        arm.set_speed_percent.assert_called_once_with(10)
        self.assertAlmostEqual(actual[0],math.radians(1))
        previous=[0]*7
        for call in arm.move_j.call_args_list:
            target=call.args[0]
            self.assertLessEqual(max(abs(a-b) for a,b in zip(target,previous)),math.radians(.1)+1e-12)
            previous=target

    def test_faster_return_does_not_relax_tracking_error(self):
        with self.assertRaisesRegex(RuntimeError,'tracking error'):
            self.simulate(follows=False,goal=[math.radians(3)]*7,speed_deg_s=2,controller_percent=10)

    def test_frozen_feedback_stops(self):
        with self.assertRaises(TimeoutError):
            self.simulate(frozen=True)

    def test_tracking_error_stops(self):
        with self.assertRaisesRegex(RuntimeError,'tracking error'):
            self.simulate(follows=False,goal=[math.radians(3)]*7)

    def test_long_return_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'60deg'):
            self.simulate(goal=[math.radians(61)]*7)

    def test_return_boundary_60_degrees(self):
        presets.check_return_span([0]*7,[math.radians(60)]*7,'leader')
        presets.check_return_span([0]*7,[-math.radians(60)]*7,'follower')
        with self.assertRaisesRegex(RuntimeError,'delta_deg'):
            presets.check_return_span([0]*7,[math.radians(60.01)]*7,'leader')

    def test_return_failure_cleans_touched_device(self):
        devices=[Mock() for _ in range(4)]
        data={'schema':1,'unit':'rad',
              'leader':{'firmware':'1.20','joints':[0]*7},
              'follower':{'firmware':'1.21','joints':[0]*7}}
        for arm in (devices[0],devices[2]):
            arm.get_config.return_value={'joint_limits':{f'joint{i}':[-2,2] for i in range(1,8)}}
        with patch('builtins.input',return_value='START'), \
                patch.object(pair,'wait_state',return_value=([1]*7,[False]*7,[])), \
                patch.object(pair,'pose',return_value=(1,[0]*7)), \
                patch.object(pair,'stable_pose',return_value=[0]*7), \
                patch.object(pair,'enable_checked',return_value=Mock()), \
                patch.object(presets,'return_to_pose',side_effect=RuntimeError('return failed')), \
                patch.object(pair,'cleanup') as cleanup:
            with self.assertRaisesRegex(RuntimeError,'return failed'):
                pair.session(*devices,5,all_joints=True,saved_pose=data)
            cleanup.assert_called_once_with({'follower':(devices[2],devices[3])},False)
        devices[0].enable.assert_not_called()


if __name__=='__main__':
    unittest.main()
