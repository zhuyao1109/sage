"""Tests for smarter tool-protocol canonicalization (keep categoricals)."""

from __future__ import annotations

import unittest

from sage_tau2.trajectory import canonicalize_tool_step, slot_argument_value


class CanonicalizeToolStepTests(unittest.TestCase):
    def test_redacts_ids_keeps_cabin(self) -> None:
        proto = canonicalize_tool_step(
            "update_reservation_flights",
            {
                "reservation_id": "XEHM4B",
                "cabin": "economy",
                "payment_id": "card_1",
                "flights": [{"flight_number": "HAT1"}],
            },
        )
        self.assertIn("cabin=economy", proto)
        self.assertIn("reservation_id=?", proto)
        self.assertIn("payment_id=?", proto)
        self.assertIn("flights=?", proto)

    def test_keeps_airport_codes(self) -> None:
        proto = canonicalize_tool_step(
            "search_direct_flight",
            {"origin": "JFK", "destination": "MCO", "date": "2024-05-01"},
        )
        self.assertIn("origin=JFK", proto)
        self.assertIn("destination=MCO", proto)
        self.assertIn("date=2024-05-01", proto)

    def test_redacts_userish_literals(self) -> None:
        self.assertEqual(slot_argument_value("user_id", "aarav_garcia_1177"), "?")
        self.assertEqual(slot_argument_value("cabin", "business"), "business")

    def test_redacts_bare_id_line_handles(self) -> None:
        self.assertEqual(slot_argument_value("id", "L1001"), "?")
        self.assertEqual(slot_argument_value("line_id", "L1001"), "?")
        self.assertEqual(slot_argument_value("customer_id", "C1001"), "?")
        proto = canonicalize_tool_step("get_details_by_id", {"id": "L1001"})
        self.assertEqual(proto, "get_details_by_id(id=?)")


if __name__ == "__main__":
    unittest.main()
