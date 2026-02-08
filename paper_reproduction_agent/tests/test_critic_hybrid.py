import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import shutil
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.critic_agent import CriticAgent

class TestCriticAgent(unittest.TestCase):
    def setUp(self):
        self.test_log_dir = "tests/test_logs"
        os.makedirs(self.test_log_dir, exist_ok=True)
        # Mock metrics tracker
        self.mock_tracker = MagicMock()
        self.mock_tracker.log_dir = self.test_log_dir

    def tearDown(self):
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir)

    def test_rule_based_blocking(self):
        """Test that rules still work and block dangerous commands."""
        critic = CriticAgent(metrics_tracker=self.mock_tracker, enable_llm_critic=False)
        
        state = {
            "current_reasoning": "I need to clone the repo",
            "proposed_action": {
                "tool_name": "execute_shell_command",
                "tool_args": {"command": "git clone https://github.com/evil/repo.git"}
            }
        }
        
        result = critic.inspect_action(state)
        self.assertFalse(result["is_authorized"])
        self.assertIn("BLOCKED", result["critic_feedback"])
        self.assertIn("Clone operations", result["critic_feedback"])

    def test_insufficient_reasoning(self):
        """Test blocking on short reasoning."""
        critic = CriticAgent(metrics_tracker=self.mock_tracker, enable_llm_critic=False)
        
        state = {
            "current_reasoning": "run", # Too short
            "proposed_action": {
                "tool_name": "execute_shell_command",
                "tool_args": {"command": "ls -la"}
            }
        }
        
        result = critic.inspect_action(state)
        self.assertFalse(result["is_authorized"])
        self.assertIn("Insufficient reasoning", result["critic_feedback"])

    @patch("src.agents.critic_agent.ChatGoogleGenerativeAI")
    def test_llm_critic_approval(self, mock_llm_class):
        """Test LLM critic approval flow."""
        # Setup mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value.content = "SAFE: This looks fine."
        mock_llm_class.return_value = mock_llm_instance
        
        critic = CriticAgent(metrics_tracker=self.mock_tracker, enable_llm_critic=True)
        
        state = {
            "current_reasoning": "I need to run this python script to analyze the data properly as requested.",
            "supervisor_directive": "Analyze the data",
            "proposed_action": {
                "tool_name": "execute_python_code", # High risk tool
                "tool_args": {"code": "print('hello')"}
            }
        }
        
        result = critic.inspect_action(state)
        
        # Should call LLM
        self.assertTrue(result["is_authorized"])
        self.assertEqual(result["inspection_type"], "llm")
        self.assertIn("Approved by LLM", result["critic_feedback"])
        self.assertEqual(critic.llm_inspections, 1)

    @patch("src.agents.critic_agent.ChatGoogleGenerativeAI")
    def test_llm_critic_blocking(self, mock_llm_class):
        """Test LLM critic blocking flow."""
        # Setup mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value.content = "DANGEROUS: This deletes system files."
        mock_llm_class.return_value = mock_llm_instance
        
        critic = CriticAgent(metrics_tracker=self.mock_tracker, enable_llm_critic=True)
        
        state = {
            "current_reasoning": "I need to analyze the system files and delete temp ones.",
            "supervisor_directive": "Optimize system",
            "proposed_action": {
                "tool_name": "execute_python_code", # High risk tool
                "tool_args": {"code": "import shutil; shutil.rmtree('/etc')"}
            }
        }
        
        result = critic.inspect_action(state)
        
        # Should be blocked
        self.assertFalse(result["is_authorized"])
        self.assertEqual(result["inspection_type"], "llm")
        self.assertIn("BLOCKED by LLM", result["critic_feedback"])
        self.assertIn("deletes system files", result["critic_feedback"])

    def test_logging(self):
        """Test that inspections are logged."""
        critic = CriticAgent(metrics_tracker=self.mock_tracker, enable_llm_critic=False)
        
        state = {
            "current_reasoning": "I need to check the files in current directory to proceed.",
            "proposed_action": {
                "tool_name": "list_directory", # Not high risk, rules only
                "tool_args": {"path": "."}
            }
        }
        
        critic.inspect_action(state)
        
        # Check log file
        log_file = os.path.join(self.test_log_dir, "critic_inspections.jsonl")
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["tool_name"], "list_directory")
            self.assertTrue(entry["is_authorized"])

if __name__ == "__main__":
    unittest.main()
