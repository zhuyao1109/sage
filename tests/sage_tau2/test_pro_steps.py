"""PRO v4 trajectory shaping for τ² (causal hist, semantic obs, think, inject)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sage_tau2.pro_steps import (
    build_booking_scratchpad,
    build_history_summary,
    build_live_window_prompt,
    messages_to_rich_pro_steps,
    parse_think_and_visible,
    runtime_inject_from_sources,
    semantic_summarize_json,
    synthesize_think,
)
from sage_tau2.runners.dump_pro_trajectories import (
    PRO_FORMAT,
    export_pro_trajectories,
    simulation_to_trajectory,
)


def _messages() -> list[dict]:
    return [
        {"role": "assistant", "content": "<think>greet</think>\nHi!"},
        {"role": "user", "content": "Change my flight."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_user_details",
                    "arguments": {"user_id": "u1"},
                }
            ],
        },
        {
            "role": "tool",
            "id": "call_1",
            "content": json.dumps(
                {
                    "user_id": "u1",
                    "name": "Ada",
                    "email": "a@x.com",
                    "reservations": [{"reservation_id": "R1"}],
                }
            ),
        },
        {
            "role": "assistant",
            "content": "<think>confirm id</think>\nFound your account.",
            "raw_data": {"sage_think": "confirm id", "sage_think_source": "model"},
        },
    ]


class ProStepsTests(unittest.TestCase):
    def test_parse_think(self) -> None:
        think, visible = parse_think_and_visible("<think>a</think>\nhello")
        self.assertEqual(think, "a")
        self.assertEqual(visible, "hello")

    def test_semantic_keeps_reservation_ids(self) -> None:
        payload = {
            "user_id": "aarav_garcia_1177",
            "name": {"first_name": "Aarav", "last_name": "Garcia"},
            "membership": "gold",
            "email": "aarav.garcia6639@example.com",
            "dob": "1992-09-13",
            "payment_methods": {
                "certificate_7473723": {"source": "certificate"},
                "gift_card_8887175": {"source": "gift_card"},
            },
            "reservations": ["M05KNL", "UHDAHF"],
        }
        summary = semantic_summarize_json(json.dumps(payload))
        self.assertIn("reservations=[M05KNL, UHDAHF]", summary)
        self.assertNotIn("reservations_n=", summary)
        self.assertIn("gift_card_8887175", summary)
        self.assertIn("name=Aarav Garcia", summary)
        self.assertIn("dob=1992-09-13", summary)

    def test_semantic_keeps_payment_last_four(self) -> None:
        payload = {
            "user_id": "daiki_muller_1116",
            "payment_methods": {
                "credit_card_2408938": {
                    "source": "credit_card",
                    "id": "credit_card_2408938",
                    "brand": "visa",
                    "last_four": "2135",
                },
                "credit_card_4303738": {
                    "source": "credit_card",
                    "id": "credit_card_4303738",
                    "brand": "visa",
                    "last_four": "5541",
                },
            },
            "reservations": ["XEHM4B"],
        }
        summary = semantic_summarize_json(json.dumps(payload))
        self.assertIn("credit_card_2408938#2135", summary)
        self.assertIn("credit_card_4303738#5541", summary)
        self.assertNotIn("email=", summary)

    def test_semantic_keeps_flight_dates(self) -> None:
        payload = {
            "reservation_id": "M05KNL",
            "origin": "ATL",
            "destination": "PHL",
            "flights": [
                {
                    "flight_number": "HAT227",
                    "origin": "ATL",
                    "destination": "ORD",
                    "date": "2024-05-23",
                    "price": 1936,
                },
                {
                    "flight_number": "HAT139",
                    "origin": "ORD",
                    "destination": "PHL",
                    "date": "2024-05-23",
                    "price": 851,
                },
            ],
            "payment_history": [
                {"payment_id": "gift_card_8887175", "amount": 100},
            ],
            "passengers": [
                {
                    "first_name": "Aarav",
                    "last_name": "Garcia",
                    "dob": "1992-09-13",
                }
            ],
        }
        summary = semantic_summarize_json(json.dumps(payload))
        self.assertIn("HAT227@05-23", summary)
        self.assertIn("HAT139@05-23", summary)
        self.assertIn("ATL>ORD", summary)
        self.assertIn("$1936", summary)
        self.assertIn("gift_card_8887175:$100", summary)
        self.assertIn("Aarav Garcia/1992-09-13", summary)

    def test_semantic_keeps_search_prices(self) -> None:
        direct = [
            {
                "flight_number": "HAT100",
                "origin": "ATL",
                "destination": "PHL",
                "date": "2024-05-24",
                "prices": {
                    "basic_economy": 68,
                    "economy": 110,
                    "business": 400,
                },
            },
            {
                "flight_number": "HAT200",
                "origin": "ATL",
                "destination": "PHL",
                "date": "2024-05-24",
                "prices": {
                    "basic_economy": 80,
                    "economy": 95,
                    "business": 350,
                },
            },
        ]
        summary = semantic_summarize_json(
            json.dumps(direct), tool_name="search_direct_flight"
        )
        self.assertIn("eco=95", summary)
        self.assertIn("eco=110", summary)
        self.assertIn("HAT200", summary)
        self.assertNotIn("list_n=", summary)
        # Compact search format: short date, no mid-option clip.
        self.assertIn("@05-24", summary)
        self.assertNotIn("...", summary)

        onestop = [
            [
                {
                    "flight_number": "HAT056",
                    "origin": "ATL",
                    "destination": "ORD",
                    "date": "2024-05-24",
                    "prices": {"economy": 100, "business": 300},
                },
                {
                    "flight_number": "HAT138",
                    "origin": "ORD",
                    "destination": "PHL",
                    "date": "2024-05-24",
                    "prices": {"economy": 80, "business": 200},
                },
            ]
        ]
        one = semantic_summarize_json(
            json.dumps(onestop), tool_name="search_onestop_flight"
        )
        self.assertIn("HAT056+HAT138", one)
        self.assertIn("eco=180", one)
        self.assertIn("ATL>ORD>PHL", one)
        self.assertNotIn("list_n=", one)

    def test_semantic_search_packs_whole_options(self) -> None:
        """8+ onestop results must not truncate mid-price (eco=2...)."""
        opts = []
        for i, eco in enumerate(
            [207, 216, 247, 250, 261, 288, 300, 320, 340, 360]
        ):
            a, b = eco * 2 // 3, eco - eco * 2 // 3
            opts.append(
                [
                    {
                        "flight_number": f"HAT{110 + i}",
                        "origin": "ATL",
                        "destination": "LGA",
                        "date": "2024-05-24",
                        "prices": {
                            "basic_economy": a - 20,
                            "economy": a,
                            "business": a * 3,
                        },
                    },
                    {
                        "flight_number": f"HAT{170 + i}",
                        "origin": "LGA",
                        "destination": "PHL",
                        "date": "2024-05-24",
                        "prices": {
                            "basic_economy": max(1, b - 10),
                            "economy": b,
                            "business": b * 3,
                        },
                    },
                ]
            )
        summary = semantic_summarize_json(
            json.dumps(opts), tool_name="search_onestop_flight"
        )
        self.assertIn("eco=207", summary)
        self.assertNotRegex(summary, r"eco=\d+\.\.\.")
        self.assertNotRegex(summary, r"eco=\d$")  # truncated digit
        # Cheapest option fully present.
        self.assertIn("HAT110+HAT170", summary)
        self.assertIn("@05-24", summary)

    def test_semantic_payment_ids_survive_history_clip(self) -> None:
        payload = {
            "user_id": "omar_rossi_1241",
            "name": {"first_name": "Omar", "last_name": "Rossi"},
            "membership": "gold",
            "email": "omar.rossi5980@example.com",
            "dob": "1970-06-06",
            "payment_methods": {
                "gift_card_8190333": {"source": "gift_card"},
                "credit_card_6754990": {"source": "credit_card"},
                "certificate_8390038": {"source": "certificate"},
                "certificate_1778167": {"source": "certificate"},
                "gift_card_6490722": {"source": "gift_card"},
                "credit_card_7407366": {"source": "credit_card"},
            },
            "reservations": ["UM3OG5", "5RJ7UH", "FQ8APE", "QKRY03"],
            "address": {"city": "X"},
            "saved_passengers": [{"first_name": "Omar"}],
        }
        summary = semantic_summarize_json(json.dumps(payload))
        self.assertIn("gift_card_8190333", summary)
        obs = f"[tool_result:get_user_details] {summary}"
        hist = build_history_summary(
            [
                {
                    "step": 1,
                    "observation_before": obs,
                    "action": "get_reservation_details(...)",
                }
            ],
            max_steps=10,
        )
        self.assertIn("gift_card_8190333", hist)
        self.assertNotIn("gift_card_819...", hist)
        self.assertNotIn("payment_methods=[gift_card_819,", hist)

    def test_booking_scratchpad_keeps_first_cabin(self) -> None:
        msgs = [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "reservation_id": "XEHM4B",
                        "cabin": "basic_economy",
                        "status": "ok",
                        "payment_history": [
                            {"payment_id": "credit_card_2408938", "amount": 296}
                        ],
                        "flights": [
                            {
                                "flight_number": "HAT005",
                                "date": "2024-05-20",
                                "price": 65,
                            }
                        ],
                    }
                ),
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "reservation_id": "XEHM4B",
                        "cabin": "business",
                        "status": "cancelled",
                        "payment_history": [
                            {"payment_id": "credit_card_2408938", "amount": 296},
                            {"payment_id": "credit_card_2408938", "amount": 1072},
                            {"payment_id": "credit_card_2408938", "amount": -296},
                            {"payment_id": "credit_card_2408938", "amount": -1072},
                        ],
                        "flights": [
                            {
                                "flight_number": "HAT005",
                                "date": "2024-05-20",
                                "price": 346,
                            }
                        ],
                    }
                ),
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "reservation_id": "7WPL39",
                        "cabin": "basic_economy",
                        "payment_history": [
                            {"payment_id": "credit_card_4303738", "amount": 402}
                        ],
                        "flights": [
                            {
                                "flight_number": "HAT246",
                                "date": "2024-05-28",
                                "price": 77,
                            }
                        ],
                    }
                ),
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "reservation_id": "A90KR2",
                        "cabin": "economy",
                        "payment_history": [
                            {"payment_id": "credit_card_4303738", "amount": 308}
                        ],
                        "flights": [
                            {
                                "flight_number": "HAT170",
                                "date": "2024-05-14",
                                "price": 154,
                            }
                        ],
                    }
                ),
            },
        ]
        pad = build_booking_scratchpad(msgs, env_date="2024-05-15")
        self.assertIn("XEHM4B", pad)
        self.assertIn("first_cabin=basic_economy", pad)
        self.assertIn("cabin=business", pad)
        self.assertIn("status=cancelled", pad)
        self.assertIn("paid=$296", pad)
        self.assertIn("7WPL39", pad)
        self.assertIn("paid=$402", pad)
        self.assertIn("upcoming_paid_sum=$698", pad)  # 296+402; A90KR2 past
        self.assertIn("upcoming", pad)
        self.assertNotIn("paid=\n", pad)
        self.assertNotRegex(pad, r"paid=\s*;")
        # Never mid-price clip on the sum suffix.
        self.assertRegex(pad, r"upcoming_paid_sum=\$\d+")
        # Pack many bookings without chopping ``paid=$``.
        long_msgs = []
        for i in range(6):
            long_msgs.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "reservation_id": f"R{i}ABCD",
                            "cabin": "economy",
                            "payment_history": [{"payment_id": "c", "amount": 100 + i}],
                            "flights": [
                                {
                                    "flight_number": "HAT1",
                                    "date": "2024-05-20",
                                    "price": 50,
                                }
                            ],
                        }
                    ),
                }
            )
        long_pad = build_booking_scratchpad(long_msgs, max_chars=200, env_date="2024-05-15")
        self.assertIn("upcoming_paid_sum=$", long_pad)
        self.assertNotRegex(long_pad, r"paid=\s*(;|$)")
        live, meta = build_live_window_prompt(
            [
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Were those basic economy?"},
            ]
            + msgs,
            history_window=2,
            task_description="Answer cabin questions.",
        )
        self.assertIn("first_cabin=basic_economy", live)
        self.assertIn("upcoming_paid_sum=", live)
        self.assertEqual(meta.get("booking_scratchpad"), pad)

    def test_clip_does_not_cut_mid_price(self) -> None:
        from sage_tau2.pro_steps import _clip_at_boundary

        text = "flights=[HAT123@05-13(SFO>PHX)$1560]; more=keep"
        clipped = _clip_at_boundary(text, 40)
        self.assertNotIn("$156...", clipped)
        self.assertNotRegex(clipped, r"\$156[^0-9]")
        # Either keep full $1560 or drop the partial token entirely.
        if "$" in clipped:
            self.assertIn("$1560", clipped)

    def test_semantic_payment_history_includes_paid_total(self) -> None:
        payload = {
            "reservation_id": "XEHM4B",
            "payment_history": [
                {"payment_id": "credit_card_2408938", "amount": 296},
                {"payment_id": "credit_card_2408938", "amount": 624},
            ],
        }
        summary = semantic_summarize_json(json.dumps(payload))
        self.assertIn("paid=$920", summary)
        self.assertIn("payment_history=", summary)

    def test_semantic_extracts_ids_from_list_of_dicts(self) -> None:
        payload = {"reservations": [{"reservation_id": "R1"}, {"reservation_id": "R2"}]}
        summary = semantic_summarize_json(json.dumps(payload))
        self.assertIn("reservations=[R1, R2]", summary)

    def test_missing_think_not_synthesized_by_default(self) -> None:
        msgs = [
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Help"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "get_user_details", "arguments": {"user_id": "u"}}
                ],
            },
            {"role": "tool", "id": "c1", "content": '{"user_id":"u"}'},
        ]
        steps = messages_to_rich_pro_steps(msgs)
        self.assertNotIn("think", steps[0])
        self.assertNotIn("think_source", steps[0])
        self.assertNotIn("observation_parts", steps[0])
        self.assertEqual(steps[0]["agent_messages"][0]["content"], "Hi!")
        self.assertNotIn("<think>", steps[0]["agent_messages"][0]["content"])
        self.assertNotIn("<think>", steps[1]["agent_messages"][0]["content"])
        synth = messages_to_rich_pro_steps(msgs, allow_synthesize_think=True)
        self.assertIn("<think>", synth[0]["agent_messages"][0]["content"])
        self.assertIn("get_user_details", synth[1]["agent_messages"][0]["content"])

    def test_causal_history_uses_observation_before(self) -> None:
        steps = messages_to_rich_pro_steps(
            _messages(), history_window=4, store_window_prompt=True
        )
        self.assertEqual(len(steps), 3)
        # Step1 agent opens: obs_before empty, after-obs is user text.
        self.assertIn(steps[0]["observation_before"], {"(none)", ""})
        self.assertIn("Change my flight", steps[0]["observation"])
        # Step2 history must pair step1 as (none)|Hi! — not user-text|Hi!
        hist1 = steps[1]["history_summary"]
        self.assertIn("[Observation 1:", hist1)
        self.assertIn("'(none)'", hist1)
        self.assertIn("Action 1: 'Hi!'", hist1)
        self.assertNotIn("Change my flight", hist1.split("Action 1:")[0])
        # Step2 conditioned on user utterance.
        self.assertIn("Change my flight", steps[1]["observation_before"])
        # Default dump omits prompt / history_summary.
        lean = messages_to_rich_pro_steps(_messages(), history_window=4)
        self.assertNotIn("prompt", lean[1])
        self.assertNotIn("history_summary", lean[1])

    def test_semantic_obs_and_tool_name(self) -> None:
        steps = messages_to_rich_pro_steps(_messages())
        obs = steps[1]["observation"]
        self.assertIn("[tool_result:get_user_details]", obs)
        self.assertIn("user_id=u1", obs)
        self.assertNotIn("reservations\":", obs)
        self.assertNotIn("observation_parts", steps[1])

    def test_full_obs_keeps_raw_tool_json(self) -> None:
        steps = messages_to_rich_pro_steps(_messages(), obs_mode="full")
        obs = steps[1]["observation"]
        self.assertEqual(steps[1].get("obs_mode"), "full")
        self.assertIn("[tool_result:get_user_details]", obs)
        self.assertIn("a@x.com", obs)
        self.assertIn('"user_id": "u1"', obs)

    def test_rich_steps_have_signals_and_no_response_bloat(self) -> None:
        steps = messages_to_rich_pro_steps(
            _messages(),
            reward_info={
                "reward": 1.0,
                "action_checks": [
                    {
                        "action": {"name": "get_user_details"},
                        "action_match": True,
                    }
                ],
            },
        )
        self.assertEqual(
            steps[0]["agent_messages"][0]["content"],
            "<think>greet</think>\nHi!",
        )
        self.assertNotIn("think", steps[0])
        self.assertEqual(steps[0]["action"], "Hi!")
        self.assertNotIn("response", steps[0])
        self.assertNotIn("observation_messages", steps[0])
        self.assertNotIn("observation_parts", steps[0])
        self.assertNotIn("prompt", steps[0])
        self.assertIn("is_action_valid", steps[0])
        self.assertEqual(steps[-1]["reward"], 1.0)
        self.assertGreaterEqual(steps[-1]["goal_progress_after"], 0.99)

    def test_live_window_prompt_matches_dump(self) -> None:
        msgs = _messages()[:2]  # after user reply, before tool call
        live, meta = build_live_window_prompt(msgs, history_window=4)
        self.assertIn("current observation", live.lower())
        self.assertIn("Change my flight", live)
        self.assertIn("'(none)'", live)
        self.assertIn("Your task is to:", live)
        self.assertNotIn("available tools", live.lower())
        self.assertEqual(meta["current_step"], 2)

    def test_prompt_stays_bounded_not_full_transcript(self) -> None:
        msgs: list[dict] = []
        for i in range(12):
            msgs.append({"role": "assistant", "content": f"say-{i}-" + ("x" * 80)})
            msgs.append({"role": "user", "content": f"user-{i}-" + ("y" * 120)})
        steps = messages_to_rich_pro_steps(
            msgs, history_window=3, store_window_prompt=True
        )
        last = steps[-1]
        self.assertLessEqual(last["history_length"], 3)
        self.assertIn("omitted", last["history_summary"])
        self.assertNotIn("say-0-", last["prompt"])
        self.assertNotIn("say-0-", last["history_summary"])
        self.assertIn("current observation", last["prompt"].lower())
        self.assertIn("Your task is to:", last["prompt"])
        self.assertNotIn("available tools", last["prompt"].lower())
        lean = messages_to_rich_pro_steps(msgs, history_window=3)
        self.assertNotIn("prompt", lean[-1])
        self.assertLessEqual(lean[-1]["history_length"], 3)

    def test_runtime_inject_and_export(self) -> None:
        sim = {
            "task_id": "16",
            "trial": 0,
            "policy": "# Airline policy",
            "termination_reason": "user_stop",
            "reward_info": {
                "reward": 1.0,
                "db_check": {"db_match": True},
                "action_checks": [],
            },
            "messages": _messages(),
        }
        dispatch = {
            "task_id": "16",
            "primary": "Executor",
            "domain": "airline",
            "layer": "eligibility_empty",
            "system_prompt": "<instructions>live</instructions>\n<policy>p</policy>",
            "organization_block": (
                "<organization>\nExecutor: Executor\nSpecialists: (none yet)\n"
                "</organization>"
            ),
            "skills_block": "",
            "injected_skill_ids": [],
            "injected_skill_names": [],
            "bank_active_count": 0,
            "bank_domain_active_count": 0,
            "task_text": "User wants to change a flight.",
        }
        inject = runtime_inject_from_sources(sim, dispatch_row=dispatch)
        self.assertEqual(inject["source"], "dispatch_journal")
        self.assertIn("live", inject["system_prompt"])
        self.assertIn("<organization>", inject["organization_block"])
        self.assertEqual(inject["skills_block"], "")

        traj = simulation_to_trajectory(
            sim, domain="airline", dispatch_row=dispatch
        )
        self.assertEqual(traj["format"], PRO_FORMAT)
        self.assertEqual(traj["task"], "User wants to change a flight.")
        self.assertEqual(traj["assigned_primary_agent"], "Executor")
        # Compact inject drops raw system_prompt body.
        self.assertNotIn("system_prompt", traj["runtime_inject"])
        self.assertEqual(traj["runtime_inject"]["system_prompt_chars"], len(dispatch["system_prompt"]))
        self.assertIn("<organization>", traj["runtime_inject"]["organization_block"])
        # Trajectory-level system once; steps reference sha1 + window prompt only.
        self.assertIn("system_prompt", traj)
        self.assertEqual(traj["system_prompt"], dispatch["system_prompt"])
        self.assertTrue(traj["steps"][0].get("prompt"))
        self.assertIn("history_summary", traj["steps"][1])
        self.assertEqual(
            traj["steps"][0].get("system_prompt_sha1"),
            traj["system_prompt_sha1"],
        )
        self.assertFalse(traj["steps"][0].get("system_in_prompt"))
        # Model think only inside agent_messages reply (ALF-style).
        self.assertEqual(
            traj["steps"][0]["agent_messages"][0]["content"],
            "<think>greet</think>\nHi!",
        )
        self.assertNotIn("think", traj["steps"][0])
        self.assertNotIn("<think>", traj["steps"][1]["agent_messages"][0]["content"])
        self.assertIn(
            "<think>confirm id</think>",
            traj["steps"][2]["agent_messages"][0]["content"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "dispatch_journal.jsonl"
            journal.write_text(json.dumps(dispatch) + "\n", encoding="utf-8")
            out = root / "pro"
            export_pro_trajectories(
                results_payload={"simulations": [sim], "info": {}},
                output_dir=out,
                domain="airline",
                dispatch_log_path=journal,
            )
            rows = [
                json.loads(line)
                for line in (out / "trajectories.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["format"], PRO_FORMAT)
            self.assertTrue((out / "runtime_context.json").exists())
            ctx = json.loads((out / "runtime_context.json").read_text())
            self.assertIn("live", ctx["shared"]["system_prompt"])
            text = (out / "task_16.txt").read_text(encoding="utf-8")
            self.assertIn("[WINDOW_PROMPT", text)
            self.assertIn("[OBSERVATION_BEFORE]", text)
            self.assertIn("[ACTION]", text)
            self.assertIn("[OBSERVATION]", text)
            self.assertIn("[REPLY]", text)
            self.assertNotIn("[THINK]", text)
            self.assertIn("[RUNTIME_INJECT]", text)
            self.assertIn("Your task is to:", text)
            # Bulk export keeps system in runtime_context, not each jsonl row.
            self.assertNotIn("system_prompt", rows[0])
            self.assertIn("prompt", rows[0]["steps"][0])
            self.assertIn("system_prompt_sha1", rows[0]["steps"][0])
            self.assertFalse(rows[0]["steps"][0].get("system_in_prompt"))
            self.assertIn("observation_before", rows[0]["steps"][0])
            # Window prompt must not inline the full system/policy body.
            self.assertNotIn("<policy>", rows[0]["steps"][0]["prompt"])
            self.assertNotIn("live", rows[0]["steps"][0]["prompt"])

    def test_synthesize_think(self) -> None:
        t = synthesize_think(
            action="Hi!",
            observation_before="(none)",
            tool_calls=None,
        )
        self.assertIn("Hi!", t)


if __name__ == "__main__":
    unittest.main()
