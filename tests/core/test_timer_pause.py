import unittest
from unittest.mock import patch

from bm25s.utils.benchmark import Timer


class TestTimerPause(unittest.TestCase):
    def test_stop_while_paused_excludes_wait_time(self):
        timer = Timer()
        with patch('bm25s.utils.benchmark.time.monotonic', side_effect=[10, 12, 22]):
            timer.start('search')
            timer.pause('search')
            self.assertEqual(timer.stop('search'), 2)
        self.assertEqual(timer.elapsed('search'), 2)
        self.assertTrue(timer.has_stopped('search'))
        self.assertFalse(timer.is_paused('search'))
        self.assertFalse(timer.is_resumed('search'))

    def test_pause_resume_tracks_state_and_multiple_intervals(self):
        timer = Timer()
        with patch('bm25s.utils.benchmark.time.monotonic', side_effect=[0, 2, 12, 15, 25, 29]):
            timer.start('search')
            self.assertFalse(timer.is_paused('search'))
            self.assertFalse(timer.is_resumed('search'))
            timer.pause('search')
            self.assertTrue(timer.is_paused('search'))
            self.assertFalse(timer.is_resumed('search'))
            timer.resume('search')
            self.assertFalse(timer.is_paused('search'))
            self.assertTrue(timer.is_resumed('search'))
            timer.pause('search')
            self.assertTrue(timer.is_paused('search'))
            self.assertFalse(timer.is_resumed('search'))
            timer.resume('search')
            self.assertEqual(timer.stop('search'), 9)
        self.assertFalse(timer.is_resumed('search'))

    def test_invalid_transitions_raise_without_reading_clock(self):
        for state in ('missing', 'running', 'paused', 'resumed', 'stopped'):
            for action in ('pause', 'resume', 'stop'):
                valid = {
                    'missing': (), 'running': ('pause', 'stop'),
                    'paused': ('resume', 'stop'), 'resumed': ('pause', 'stop'),
                    'stopped': (),
                }
                if action in valid[state]:
                    continue
                with self.subTest(state=state, action=action):
                    timer = Timer()
                    with patch('bm25s.utils.benchmark.time.monotonic', return_value=1):
                        if state != 'missing':
                            timer.start('search')
                        if state in ('paused', 'resumed'):
                            timer.pause('search')
                        if state == 'resumed':
                            timer.resume('search')
                        if state == 'stopped':
                            timer.stop('search')
                    before = timer.to_dict()
                    with patch('bm25s.utils.benchmark.time.monotonic') as clock:
                        with self.assertRaises(ValueError):
                            getattr(timer, action)('search')
                        clock.assert_not_called()
                    self.assertEqual(timer.to_dict(), before)

    def test_unpaused_timer_and_export_are_unchanged(self):
        timer = Timer(precision=2)
        with patch('bm25s.utils.benchmark.time.monotonic', side_effect=[10, 12.345]):
            self.assertEqual(timer.start('Search Query'), 'Search Query')
            self.assertAlmostEqual(timer.stop('Search Query'), 2.345)
        self.assertEqual(timer.elapsed('Search Query'), 2.35)
        self.assertEqual(timer.to_dict(), {
            'Search Query': {'start': 10, 'elapsed': 2.3450000000000006, 'stopped': 12.345}
        })
        self.assertEqual(list(timer.to_dict(underscore=True, lowercase=True)), ['search_query'])
        exported = timer.to_dict()
        exported['Search Query']['elapsed'] = -1
        self.assertEqual(timer.elapsed('Search Query'), 2.35)

    def test_independent_timers_do_not_share_pause_state(self):
        timer = Timer()
        with patch('bm25s.utils.benchmark.time.monotonic', side_effect=[0, 1, 2, 4, 5]):
            timer.start('index')
            timer.start('query')
            timer.pause('index')
            self.assertFalse(timer.is_paused('query'))
            self.assertEqual(timer.stop('query'), 3)
            self.assertEqual(timer.stop('index'), 2)


if __name__ == '__main__':
    unittest.main()
