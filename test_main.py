import unittest
import os
from main import run_simulation

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.output_dir = "C:/school/programare/LICENTA/licenta"
        self.results_dir = os.path.join(self.output_dir, "results")
        self.temp_dir = os.path.join(self.output_dir, "temp")
        self.gif_path = os.path.join(self.results_dir, 'robot_simulation.gif')

    def tearDown(self):
        # Clean up created files and directories (optional)
        pass

    # def test_run_simulation(self):
    #     run_simulation()
    #     self.assertTrue(os.path.exists(self.gif_path), "GIF file was not created")

    def test_robots_reach_goals(self):
        run_simulation()
        log_path = os.path.join(self.results_dir, 'simulation_log.txt')
        self.assertTrue(os.path.exists(log_path), "Log file was not created")
        with open(log_path, 'r') as log_file:
            log_content = log_file.read()
        self.assertIn("All robots reached their goals!", log_content)

if __name__ == '__main__':
    unittest.main()