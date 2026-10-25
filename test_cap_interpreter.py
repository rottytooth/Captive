import unittest
from cap_interpreter import interpreter


class TestCapInterpreter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the interpreter once for all tests."""
        print("Initializing interpreter for tests...")
        cls.interpreter = interpreter()
        print("Interpreter initialized.")

    def test_hello_world(self):
        """Test that hello_world.txt outputs 'Hello, World!'"""
        result = self.interpreter.interpret_file('programs/hello_world.txt')

        self.assertEqual(
            result,
            "Hello, World!",
            f"Expected 'Hello, World!', but got '{result}'"
        )

    def test_hi1(self):
        """Test that hi_basic.txt outputs 'Hi'"""
        result = self.interpreter.interpret_file('programs/hi_basic.txt')

        self.assertEqual(
            result,
            "Hi",
            f"Expected 'Hi', but got '{result}'"
        )

    def test_hi2(self):
        """Test that hi.txt outputs 'Hi'"""
        result = self.interpreter.interpret_file('programs/hi.txt')

        self.assertEqual(
            result,
            "Hi",
            f"Expected 'Hi', but got '{result}'"
        )

    def run_text(self, text):
        return self.interpreter.parse(self.interpreter.lexer.lex(text))

    def test_if_on_empty_stack(self):
        """An if with nothing on the stack is false rather than a crash"""
        self.assertEqual(self.run_text("pot"), ("", []))

    def test_while_on_empty_stack(self):
        """A while with nothing on the stack is false rather than a crash"""
        self.assertEqual(self.run_text("go"), ("", []))

    def test_not(self):
        """not turns 0 into 1 and negates anything else"""
        self.assertEqual(self.interpreter.parse(["push", 0, "not"]), ("", [1]))
        self.assertEqual(self.interpreter.parse(["push", 5, "not"]), ("", [-5]))

    def test_nested_block_in_skipped_if(self):
        """A block nested inside a skipped if needs its own end before the outer one closes"""
        nodes = ["push", 66, "push", 0, "if", "if", "end", "push", 65, "emit", "end", "pop", 1, "emit"]
        self.assertEqual(self.interpreter.parse(nodes), ("B", []))

    def test_if_inside_while(self):
        """The end of an if inside a while closes only the if, not the loop"""
        nodes = ["push", 0, "push", 66, "push", 65,
                 "while", "dup", "if", "emit", "end", "emit", "end"]
        self.assertEqual(self.interpreter.parse(nodes), ("AABB", [0]))

    def test_nested_while(self):
        """An inner loop doesn't wipe the outer loop's history, so the outer loop's cycle is still detected"""
        # the outer loop rotates through three states forever; each pass runs an inner loop once
        nodes = ["push", 1, "push", 2, "push", 3,
                 "while", "rot", 1, "push", 1, "while", "push", -1, "add", "end", "pop", 3, "end"]
        self.assertEqual(self.interpreter.parse(nodes), ("", [1, 2, 3]))

    def test_final_stack_after_rot(self):
        """The returned stack reflects a rot at the top level"""
        self.assertEqual(self.interpreter.parse(["push", 1, "push", 2, "rot", 1]), ("", [2, 1]))


if __name__ == "__main__":
    unittest.main()
