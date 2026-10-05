import unittest

from app.scoring import cosine_distance_to_similarity


class CosineDistanceToSimilarityTests(unittest.TestCase):
    def test_identical_vectors_have_full_similarity(self):
        self.assertEqual(cosine_distance_to_similarity(0.0), 1.0)

    def test_opposite_vectors_have_negative_similarity(self):
        self.assertEqual(cosine_distance_to_similarity(2.0), -1.0)

    def test_result_is_rounded_to_two_decimal_places(self):
        self.assertEqual(cosine_distance_to_similarity(0.456), 0.54)

    def test_values_are_not_clamped(self):
        self.assertEqual(cosine_distance_to_similarity(-0.25), 1.25)


if __name__ == "__main__":
    unittest.main()
