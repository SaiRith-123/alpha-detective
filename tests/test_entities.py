from __future__ import annotations

import unittest

from llama_index.core import Document

from alpha_detective import EntityIndex


def make_node(company: str, ticker: str, quarter: str = "2020-Q1") -> Document:
    return Document(
        text="Transcript evidence.",
        metadata={
            "Company_Name": company,
            "Ticker": ticker,
            "Quarter": quarter,
        },
    )


class DetectEntitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entity_index = EntityIndex(
            [
                make_node("Apple Inc.", "A"),
                make_node("Microsoft Corporation", "MSFT"),
                make_node("Acme Co.", "ACME"),
            ]
        )

    def test_company_name_match_strips_corporate_suffix(self) -> None:
        self.assertEqual(
            self.entity_index.detect_entities("Tell me about Apple"),
            (["A"], []),
        )

    def test_uppercase_ticker_match(self) -> None:
        self.assertEqual(
            self.entity_index.detect_entities("How did MSFT perform?"),
            (["MSFT"], []),
        )

    def test_lowercase_single_letter_does_not_match_ticker(self) -> None:
        self.assertEqual(
            self.entity_index.detect_entities("What happened to a?"),
            ([], []),
        )

    def test_multi_company_query(self) -> None:
        self.assertEqual(
            self.entity_index.detect_entities("Compare Apple and Microsoft"),
            (["A", "MSFT"], []),
        )

    def test_no_entity(self) -> None:
        self.assertEqual(
            self.entity_index.detect_entities("What was the market outlook?"),
            ([], []),
        )

    def test_quarter_parse(self) -> None:
        self.assertEqual(
            self.entity_index.detect_entities("Compare Apple in 2020-q1 and 2022-Q4"),
            (["A"], ["2020-Q1", "2022-Q4"]),
        )


if __name__ == "__main__":
    unittest.main()
