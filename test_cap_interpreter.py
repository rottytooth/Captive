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

    def test_fibonacci(self):
        """fibonacci_basic.txt prints the first 24 Fibonacci numbers, and fibonacci.txt spells the same program"""
        fib = [1, 1]
        while len(fib) < 24:
            fib.append(fib[-1] + fib[-2])
        self.assertEqual(self.interpreter.interpret_file('programs/fibonacci_basic.txt'), ", ".join(map(str, fib)))
        with open('programs/fibonacci.txt', encoding='utf-8') as text, open('programs/fibonacci_basic.txt', encoding='utf-8') as basic:
            self.assertEqual(self.interpreter.lexer.lex(text.read()), self.interpreter.lexer.lex(basic.read()))

    def run_text(self, text):
        return self.interpreter.parse(self.interpreter.lexer.lex(text))

    def test_if_on_empty_stack(self):
        """An if with nothing on the stack is false rather than a crash"""
        self.assertEqual(self.run_text("pot"), ("", []))

    def test_while_on_empty_stack(self):
        """A while with nothing on the stack is false rather than a crash"""
        self.assertEqual(self.run_text("go"), ("", []))

    def test_sub(self):
        """f subtracts: push 1, push 73, sub leaves 72 ('H')"""
        self.assertEqual(self.run_text("ltyldptyft"), ("H", []))

    def test_sub_in_prose(self):
        """The f in an ordinary word like "of" is a sub command"""
        self.assertEqual(self.run_text("ltyldpty of to"), ("H", []))

    def test_mod(self):
        """q is mod, top % second: push 3, push 10, mod leaves 1"""
        self.assertEqual(self.run_text("ljtylpttyq"), ("", [1]))

    def test_mod_by_zero(self):
        """Like div, mod by zero gives 0"""
        self.assertEqual(self.run_text("lylpttyq"), ("", [0]))

    def test_pop_reads_its_index(self):
        """pl is followed by the index to pop, with no stray end: push 1, push 4, pop index 1 leaves [1]"""
        self.assertEqual(self.interpreter.lexer.lex("ltylgyplty"), ["push", 1, "push", 4, "pop", 1])
        self.assertEqual(self.run_text("ltylgyplty"), ("", [1]))

    def test_rot_by_one(self):
        """pp rotates the top item to the bottom and dp the bottom item to the top, with no number"""
        self.assertEqual(self.interpreter.lexer.lex("ltylgylpypp"), ["push", 1, "push", 4, "push", 8, "rot", 1])
        self.assertEqual(self.run_text("ltylgylpypp"), ("", [8, 1, 4]))
        self.assertEqual(self.interpreter.lexer.lex("ltylgylpydp"), ["push", 1, "push", 4, "push", 8, "rot", -1])
        self.assertEqual(self.run_text("ltylgylpydp"), ("", [4, 8, 1]))

    def test_rot_reads_its_count(self):
        """pg is followed by the count to rotate"""
        self.assertEqual(self.interpreter.lexer.lex("ltylgylpypgjy"), ["push", 1, "push", 4, "push", 8, "rot", 2])
        self.assertEqual(self.run_text("ltylgylpypgjy"), ("", [4, 8, 1]))

    def test_swap_and_over(self):
        """th swaps the top two items, and dt copies the second to the top"""
        self.assertEqual(self.run_text("ltyljyth"), ("", [2, 1]))
        self.assertEqual(self.run_text("ltyljydt"), ("", [1, 2, 1]))

    def test_t_alone_is_emit(self):
        """t is emit unless an h follows, and a dropped d keeps an emit and a dup apart"""
        self.assertEqual(self.interpreter.lexer.lex("tt"), ["emit", "emit"])
        self.assertEqual(self.interpreter.lexer.lex("th"), ["swap"])
        self.assertEqual(self.interpreter.lexer.lex("tdh"), ["emit", "dup"])

    def test_unfinished_pair_is_dropped(self):
        """A p or d that begins no command is dropped, and the next letter read on its own"""
        self.assertEqual(self.interpreter.lexer.lex("pk"), ["div"])
        self.assertEqual(self.interpreter.lexer.lex("dh"), ["dup"])

    def test_binary_numbers(self):
        """Number letters are binary places, d 64 down to t 1"""
        self.assertEqual(self.interpreter.lexer.lex("ldhlpgjty"), ["push", 127])
        self.assertEqual(self.interpreter.lexer.lex("ldy lhy lly lpy lgy ljy lty"),
                         ["push", 64, "push", 32, "push", 16, "push", 8, "push", 4, "push", 2, "push", 1])

    def test_f_subtracts_in_number(self):
        """Inside a number, f subtracts the next value: d f t is 64 - 1 = 63 ('?')"""
        self.assertEqual(self.run_text("ldftyt"), ("?", []))

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
