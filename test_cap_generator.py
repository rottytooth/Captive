import unittest
from cap_generator import AdvancedSpecialLetterGenerator


class TestAdvancedSpecialLetterGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the generator once for all tests."""
        print("Initializing generator for tests...")
        cls.generator = AdvancedSpecialLetterGenerator()
        print("Generator initialized.")

    def extract_special_letters(self, text, special_letters):
        """Extract all special letters from text in order."""
        return ''.join(c for c in text.lower() if c in special_letters)

    def test_special_letters_match_input(self):
        """Test that the special letters in output match exactly the input sequence."""
        test_cases = [
            "bddg",
            "kt",
            "hpy",
            "ldgptyldpytttt",
            "fg",
            "plt"
        ]

        for test_input in test_cases:
            with self.subTest(input=test_input):
                phrases = self.generator.generate_grammatical_phrases(test_input, max_results=3)

                # Should generate at least one phrase
                self.assertGreater(len(phrases), 0, f"No phrases generated for input '{test_input}'")

                for phrase in phrases:
                    # Extract special letters from the generated phrase
                    output_special = self.extract_special_letters(phrase, self.generator.special_letters)

                    # The special letters in output should match input exactly
                    self.assertEqual(
                        output_special,
                        test_input,
                        f"Special letters mismatch for input '{test_input}'. "
                        f"Expected: '{test_input}', Got: '{output_special}' in phrase '{phrase}'"
                    )

    def test_ldgptyldpytttt(self):
        """Test that the special letters in output match exactly for 'ldgptyldpytttt'."""
        test_input = "ldgptyldpytttt"
        phrases = self.generator.generate_grammatical_phrases(test_input, max_results=3)

        self.assertGreater(len(phrases), 0, f"No phrases generated for input '{test_input}'")

        for phrase in phrases:
            output_special = self.extract_special_letters(phrase, self.generator.special_letters)

            self.assertEqual(
                output_special,
                test_input,
                f"Special letters mismatch for input '{test_input}'. "
                f"Expected: '{test_input}', Got: '{output_special}' in phrase '{phrase}'"
            )

    def test_no_extra_special_letters(self):
        """Test that no special letters appear that weren't in the input."""
        test_cases = [
            "bddg",
            "kt",
            "hpy",
            "fg"
        ]

        for test_input in test_cases:
            with self.subTest(input=test_input):
                phrases = self.generator.generate_grammatical_phrases(test_input, max_results=3)

                for phrase in phrases:
                    # Get the set of special letters that appear in the input
                    input_special_set = set(test_input)

                    # Get all special letters from the output
                    output_special = self.extract_special_letters(phrase, self.generator.special_letters)
                    output_special_set = set(output_special)

                    # Every special letter in output must have been in input
                    self.assertTrue(
                        output_special_set.issubset(input_special_set),
                        f"Output contains special letters not in input. "
                        f"Input letters: {input_special_set}, "
                        f"Output letters: {output_special_set}, "
                        f"Phrase: '{phrase}'"
                    )

    def test_all_output_letters_are_valid(self):
        """Test that all letters in output are either special or non-special allowed letters."""
        test_input = "bddg"
        phrases = self.generator.generate_grammatical_phrases(test_input, max_results=3)

        all_allowed_letters = self.generator.special_letters | self.generator.non_special_letters

        for phrase in phrases:
            for char in phrase.lower():
                if char.isalpha():
                    self.assertIn(
                        char,
                        all_allowed_letters,
                        f"Invalid letter '{char}' found in phrase '{phrase}'"
                    )

    def test_long_sequence(self):
        """Test that long sequences are properly handled."""
        test_input = "lgtyldglyldgplylddbhttyldgpllbtyldjktylgylgplyldghbtyldpglyldgplyldgltyldpygty"
        phrases = self.generator.generate_grammatical_phrases(test_input, max_results=2)

        # Should generate at least one phrase
        self.assertGreater(len(phrases), 0, "No phrases generated for long input")

        for phrase in phrases:
            output_special = self.extract_special_letters(phrase, self.generator.special_letters)

            # The special letters in output should match input exactly
            self.assertEqual(
                output_special,
                test_input,
                f"Special letters mismatch for long input. "
                f"Expected length: {len(test_input)}, Got length: {len(output_special)}"
            )

    def test_phrase_formatting(self):
        """Test that generated phrases are properly formatted."""
        test_input = "bddg"
        phrases = self.generator.generate_grammatical_phrases(test_input, max_results=3)

        for phrase in phrases:
            # Should start with capital letter
            self.assertTrue(phrase[0].isupper(), f"Phrase doesn't start with capital: '{phrase}'")

            # Should end with period
            self.assertTrue(phrase.endswith('.'), f"Phrase doesn't end with period: '{phrase}'")

            # Should not be empty
            self.assertGreater(len(phrase), 2, f"Phrase too short: '{phrase}'")


if __name__ == "__main__":
    unittest.main()
