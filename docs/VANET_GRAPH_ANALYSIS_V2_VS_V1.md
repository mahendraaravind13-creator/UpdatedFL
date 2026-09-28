# Reading the VANET graphs: BC-PAFL v2 vs ProxyFL v1

This walks through the 17 VANET figures from a 100-round BC-PAFL v2 run and compares
each one with the matching ProxyFL v1 figure. For every graph it covers what is plotted,
why the line has the shape it does, and why it differs from v1.

**The v2 run:** 100 rounds, 28 vehicles, 5 RSUs, 1 server (3,400 metric rows), seed 42,
on `master` after PR #1 (commit `260a8b8`):

```
python main.py --rounds 100 --vehicles 28 --output results/final2
```

Outputs land in `results/final2/pomdp/` (`vanet_*.png`, `vanet_plot_explanations.md`,
`vanet_metrics.csv`, `vanet_routing_rounds.csv`, `vanet_routing_metadata.json`). The
`results/` folder is gitignored, so rerun the command to see the figures.

**The v1 reference:** ProxyFL v1's `plots/` folder (100 rounds, 21 vehicles, 5 RSUs). All
v1 numbers here come from its PNGs, its `vanet_plot_explanations.md` and its current
`vanet_metrics.csv` / `vanet_routing_rounds.csv`. Those agree with each other and show a
healthy run with no failed route discoveries.

One caution: v1's `plots/vanet_graph_analysis.md` describes a *different* run, one where
coverage collapsed to zero after round 43, NRL hit 26.4 and latencies reached 34 s. It
doesn't match v1's current CSVs or images, so none of its numbers are used here.

## The setup explains most of the differences

| | v2 (`final2`) | v1 reference |
|---|---|---|
| Vehicles / rounds / RSUs | 28 / 100 / 5 | 21 / 100 / 5 |
| Data split | IID (even) | non-IID, Dirichlet α = 0.5 |
| Shared model | 4→64→64→6 MLP (~4.8k params) | larger models, 3 local epochs |
| Local training | 1–2 epochs (set by the POMDP), LR × 0.95 per round | fixed LR |
| Private models | all 28 train every round | all 21 train every round |
| Mobility | 10–30 m/s, bounce off the boundary, handover | 2–8 m/s random walk, no re-entry |
| Round length | 90 simulated seconds | 10 simulated seconds |
| Transport | real single-hop V2I, with AODV as a modeled overlay | live AODV multi-hop |
| Link cap | 6 Mbps | 27 Mbps |
| Crypto | MIRACL P-256 + C-backed AES-GCM, dynamic pseudonyms | costlier per operation, static identities |
| Energy constants | 10.88 W; x = 1.0 / 0.4 / 0.6 / 0.2 | same |

## The short version

IID data and a decaying learning rate make v2 converge quickly (rounds 1–10), after
which the fleet stays tight: private accuracy 81–82%, test accuracy 82–83%. The POMDP
picks 3–11 vehicles per round with 20–40% dropouts, and global accuracy on the attack
benchmark climbs to about 87% and stays there. Two malicious updates got through before
the TA revoked them (rounds 17 and 52).

Three things that went wrong in v1's run don't happen here:

- **Coverage never collapses**, because v2 vehicles drive back into range.
- **Routing never fails**: the overlay has zero `no_route` outcomes in all 100 rounds.
- **The global model never freezes**: at least one update is accepted every round.

## 1. Accuracy vs rounds

`vanet_accuracy_vs_rounds.png`

**What's plotted.** 28 faint lines, one per vehicle's private model, scored on its own
held-out data. The thick green line is their mean. Between trainings a vehicle's last
value is carried forward, which is exact because its private weights can't change while
it isn't training. The blue line is the global proxy model scored on the **attack
benchmark** (`attack1-5_test.csv`), which is what v1's server reports. Accuracy on the
normal test split is logged separately as `global_test_accuracy_pct`.

**Why this shape.** The private mean starts at 80.8% (R1), peaks at 81.7% (R53) and ends
around 81.6%. It's flat because IID data and LR decay settle the fleet early. By R100 the
spread is only 79.3–84.2%. Global attack accuracy jumps from 71.8% (R1) to 86.6% (R50)
and holds there to R100, peaking at 87.2%. Test accuracy goes from 80.6% to 82.5% to
82.4%. The only steep part is R1–R10, when the learning rate is still high and the
biggest errors get fixed first.

**Compared with v1.** v1's private mean rises from 79.5% (R1) to 82.5% (R47–99), with a
77–84% spread. Its global line (held-out *test* data, not attacks) goes from 60.0% to a
75.6–76.4% plateau.

- **The blue lines measure different things.** v2's is attack-only accuracy, with no
  benign majority to lean on; v1's is overall test accuracy. Compare v2's test line
  (80.6% → 82.4%) with v1's 76.4% instead. v2 wins that because of its bigger shared
  model and IID data.
- **v1 keeps climbing and v2 flattens.** Non-IID data keeps giving v1 something new to
  learn, while IID data plus LR decay settles v2.
- **v1 starts much lower.** Its 60% start reflects a weak initialisation; v2 is already
  at 71.8% / 80.6% after one FL round.
- **No v2 vehicle gets stuck low anymore.** Before PR #1, vehicles V013 and V018 sat
  around 37%: they had attack-heavy partitions and only 1–2 epochs to move away from a
  majority-biased global model. IID data and continuous private training fixed that.

## 2. Training loss vs rounds

`vanet_loss_vs_rounds.png`

**What's plotted.** Each vehicle's **private-model DML loss**, (1 − α)·CE + α·KL, which
is v1's definition, plus the mean. Loss only exists while training, so nothing is carried
forward.

**Why this shape.** The mean goes from 0.59 (R1) to a minimum of 0.36 (R76) and ends
near 0.37. The drop over R1–R2 is the steepest, since the largest errors get corrected
while the LR is highest. After that there's a long tail with a ±0.01 sawtooth from
minibatches, the moving DML target and the shrinking step size.

**Compared with v1.** v1 goes from 0.591 to 0.368, practically the same curve. Both
fleets fit small local datasets with DML under the same batch dynamics. v1 logs the
proxy-side loss rather than the private one, but the matching shape confirms the
training loops behave the same.

## 3. Training energy vs rounds

`vanet_energy_training_vs_rounds.png`

**What's plotted.** Mean per-vehicle `E = 10.88 × 1.0 × training_time`. It ranges from
2.39 J (R35) to 4.38 J (R5), averaging 2.84 J, and follows training time exactly: mean
training took about 402 ms in R5 and about 220 ms in R35.

**Compared with v1.** v1 uses 840–1274 J (mean 941 J), roughly 330× more. The formula
and constants are identical; the gap is compute. v1 trains much larger models for about
86 s per vehicle per round on a shared GPU, while v2 trains a 4.8k-parameter MLP for
about 261 ms. That's why v2's energy fits in joules instead of kilojoules. Ups and downs
follow batch scheduling and active sample counts, not accuracy.

## 4. Non-training energy vs rounds

`vanet_energy_other_vs_rounds.png`

**What's plotted.** Security energy (10.88 × 0.4 × crypto time) plus communication
energy (10.88 × 0.6 × airtime), logged as `E_total`. It ranges from 0.16 J (R24) to
1.12 J (R1), averaging 0.38 J. It drops after R1 once the burst of pseudonym
provisioning and the large early batches are over.

**Compared with v1.** v1 uses 2.0–13.7 J (mean 5.4 J), about 14× more. The accounting is
the same, but v1's operations cost far more. Its decryption averages 367 ms against v2's
0.08 ms, and verification 215 ms against 52 ms. Its TCP exchanges also take much longer
than v2's modeled airtime (mean 419 ms vs 32 ms).

## 5. Energy breakdown

`vanet_energy_breakdown.png`

The three non-training series plus training energy. v2's training energy (about 2.8 J)
is only about 7× the overhead total (about 0.38 J), so all four lines stay readable. In
v1, 724–965 J of training squashes the 0.26–2.96 J overhead lines flat against the axis.
The visual difference is purely scale and follows directly from section 3: v2's model
is three orders of magnitude cheaper to train per round.

## 6. End-to-end time vs rounds

`vanet_end_to_end_time_vs_rounds.png`

**What's plotted.** Per vehicle, `training + security + communication + idle`. The mean
is 2776 ms, ranging from 1064 ms (R24) to 9818 ms (R1). Average components: training
261 ms, security 40 ms, communication 32 ms, idle 2442 ms. Idle time is the round's
wall-clock minus everything active, and it dominates: vehicles spend most of each round
waiting for the slowest participant and for the round to close, not computing.

**Compared with v1.** v1 takes 105–169 s (mean 115 s): about 86 s of training and 28 s
idle. v1's training time swamps everything, whereas v2 trains in 0.26 s, which leaves
idle as the only big term. Both peak in R1 (cold caches, full pools, biggest batches)
and then settle. The roughly 40× axis difference comes from model size and threading,
not a measurement artifact.

## 7. Latency breakdown

`vanet_latency_breakdown.png`

Same series and scale as the end-to-end figure, kept under its older name. As in v1,
it isn't a separate experiment, so don't count it twice.

## 8. Security and communication detail

`vanet_end_to_end_security_comm_detail.png`

**What's plotted.** Security ranges from 11.2 ms (R24) to 172.1 ms (R7), averaging
40.4 ms. Communication ranges from 14.6 ms (R4) to 69.0 ms (R1), averaging 31.9 ms.
Security spikes line up with rounds that had big verification batches and bursts of
pseudonym provisioning (R1, R7). Communication tracks total model bytes and link
distance.

**Compared with v1.** v1's security runs 283–1714 ms (mean 621 ms) and communication
98–1186 ms (mean 419 ms). v1 times real TCP send/receive threads on a much heavier
crypto stack. v2's communication is modeled airtime for messages of about 19 KB (around
25 ms at up to 6 Mbps), and its crypto is C-backed with a verification cache. Both
decline and then level off as the protocol has less work to do. That doesn't mean the
network got faster.

## 9. Action-to-response latency

`vanet_action_to_response_latency.png`

**What's plotted.** For each selected vehicle, simulated time from sending TRAIN_CONFIG
to finishing its upload. It ranges from 18.9 s (R38) to 51.8 s (R49), averaging 34.7 s.
The 90 s round structure dominates: local compute plus waiting for a simulated upload
slot. So the line moves with which fast or slow vehicles the POMDP picked, not with
network quality.

**Compared with v1.** v1 shows 4.6–13.8 s (mean 8.5 s), but that's a different timer:
wall-clock time from sending LOCAL_UPDATE to receiving GLOBAL_UPDATE. **Don't read the
gap as a latency win for either side.** v2's number can include up to a full round of
simulated compute and scheduling; v1's is a host round-trip.

## 10. Cryptographic operations

`vanet_cryptographic_operations.png`

**What's plotted (means).** Signature generation 13.4–30.2 ms (15.5 ms); batch
verification about 29 ms of a roughly 52 ms verification total; AES-GCM encryption
8.9–43.2 ms (21.2 ms). Every outgoing message is signed and encrypted in all 100 rounds,
so those two stay in a narrow band. Verification follows batch size and peaks in R2 and
R7, which had the largest beacon and upload batches.

**Compared with v1.** v1's generation is 17–150 ms (51 ms), verification 145–397 ms
(215 ms) and encryption 80–323 ms (140 ms), roughly 3–7× higher across the board. v2
wins through smaller batches, a verification cache shared across co-hosted replicas, and
C-backed AES-GCM (decryption 0.08 ms vs 367 ms).

v2 does pay one cost that v1 doesn't: generating pseudonym keys takes about 212 ms per
provisioned vehicle per round (MIRACL EC key generation for the whole pool). That's the
price of dynamic pseudonyms over v1's fixed identities. It's recorded on the provisioned
vehicles' rows and included in their security energy.

## 11. Security overhead

`vanet_security_overhead.png`

The same four series as section 10, kept under its older name, as in v1. It reads the
same way and isn't separate evidence.

## 12. Link goodput vs rounds

`vanet_throughput_vs_rounds.png`

**What's plotted.** `sum(delivered wireless bits) / sum(modeled airtime)` per round, over
the AODV overlay's successful hops. The result is **exactly 6.00 Mbps in all 100
rounds**. That's not a bug: every overlay link in this run is short enough to hit v2's
6 Mbps cap, so the ratio equals the cap by construction.

**Compared with v1.** v1 falls from 11.28 Mbps (R1) to 2.48 Mbps (R71), averaging 5.2.
It's a real curve because v1's 27 Mbps cap rarely binds, and as coverage collapses its
traffic shifts onto long, weak links. The differences come from:

- the cap itself (6 vs 27 Mbps);
- v2's traffic staying on short links, thanks to a dense 28-vehicle mesh, nearest-RSU
  destinations and coverage that never collapses;
- v2 only counting delivered-hop airtime at the capped rate.

The honest limitation: this run never uses a link below capacity, so the figure confirms
the accounting works but tells you nothing about link adaptation. Also, this isn't FL or
server throughput; that's `throughput_bytes_per_sec` on the Server rows.

## 13. Vehicles in RSU range

`vanet_vehicles_in_range_vs_rounds.png`

**What's plotted.** How many vehicles have each RSU as their closest in-range RSU, plus
the total. The total goes from 26 (R1) to 20 (R100), dipping to 10 in several mid-run
rounds. Per RSU at R1 / R50 / R100: RSU_0 3/7/7, RSU_1 7/2/4, RSU_2 7/4/3, RSU_3 5/4/2,
RSU_4 4/1/4. Every RSU serves at least one vehicle in every round.

**Compared with v1.** v1 is a staircase down from 21 to 0 and stays at exactly 0 from
R43 to R100. Its vehicles crawl at 2–8 m/s with no handover and never turn back at the
boundary, so once they leave they're gone. v2 vehicles drive at 10–30 m/s and bounce off
the boundary back into coverage, so the total wobbles instead of decaying. You can see
the knock-on effect everywhere else: v2's server accepts at least one RSU model every
round, whereas v1's global accuracy goes perfectly flat after R44.

## 14. AODV routing overhead

`vanet_aodv_routing_overhead_vs_rounds.png`

**What's plotted.** RREQ (24 B), RREP (20 B) and RERR message bodies, plus 28 B of IP/UDP
header per transmission. On average there are about 55 RREQs, 16 RREPs and **zero RERRs**
per round, about 3.5 KiB in total. RERR is zero because the modeled links never break
mid-message; routes are only rediscovered when they expire.

**Compared with v1.** v1 averages about 109 RREQs, 24 RREPs and almost no RERRs per
round: 9.4–13.7 KiB early on, and up to 28.2 KiB during the flooding spell in R28–R34.
v2 is cheaper because its destinations are the nearest RSUs across a dense mesh, usually
1–2 hops away and stable for the whole 90 s round. v1's paths keep getting longer as
coverage collapses. Packet sizes and flood-on-expiry rules are the same in both.

## 15. Wireless communication volume

`vanet_communication_volume_vs_rounds.png`

**What's plotted.** Bytes over the air, counted per hop at the IP layer. The mean is
878 KiB per round: 623 FL, 231 security, 1.6 routing bodies and 23 headers. A multi-hop
payload counts once for each hop. It stays high all run (562–1573 KiB) because
deliveries never dry up.

**Compared with v1.** v1 averages 256 KiB (FL 194, security 50, routing 3, headers 10),
peaking at 393 KiB in R11 and falling as deliveries stop. v2 sends about 3.4× more
bytes because each signed and encrypted model message is about 19 KB: 4.8k float32
parameters plus signatures and AEAD overhead, split into 1200 B DATA packets. v1's
payloads are smaller. The right reading is that v2 moves more data and delivers far more
reliably, not that it's less efficient. The next graph shows that.

## 16. Normalized Routing Load

`vanet_normalized_routing_load_vs_rounds.png`

**What's plotted.** `(RREQ + RREP + RERR transmissions) / data packets delivered`,
ranging from 0.048 to 0.237 (mean 0.09). It's defined in every round because there are
no `no_route` outcomes, so no "undefined" markers appear.

**Compared with v1.** v1's healthy run sits at 0.36–1.06 (mean 0.54); the collapsed run
from the old analysis doc reached 26.4. v2's denominator is huge, since fragmented
19 KB messages crossing several hops add up to thousands of delivered DATA packets per
round. Its numerator stays small because route discoveries are short. v1 sends smaller
messages over longer paths with more flooding. NRL is an efficiency ratio rather than a
byte count, so v2 can have a low NRL and high absolute volume (section 15) at the same
time.

## 17. AODV modeled network latency

`vanet_aodv_network_latency_vs_rounds.png`

**What's plotted.** Mean simulated delivery time per envelope. The successful-only and
all-attempts lines **sit on top of each other** at 0.61–1.13 s (mean 0.83 s) because
every envelope was delivered. With no failed discoveries, none of the 36 s no-route
timeouts get into the average.

**Compared with v1.** v1 sits at 0.288–0.314 s, with its two lines also overlapping. v2
is about 3× slower per envelope because its bigger messages split into more 1200 B DATA
packets, sent one after another on each hop at up to 6 Mbps, while v1 sends smaller
messages at up to 27 Mbps. This is simulated wireless time and never slows down the
real run. If any discovery had failed, the all-attempts line would step upward the way
it does in v1's collapsed run.

## What to take away

- **Private models keep learning on every vehicle.** All 28 report every round and the
  R100 spread is 5 points. The pre-PR collapse is gone, which confirms its diagnosis:
  attack-heavy partitions plus 1–2 epochs from a majority-biased global model, fixed by
  IID data and continuous private DML.
- **Lower late-run security and communication costs mean the protocol settled down, not
  that the network got faster.** The same goes for the flat 6.0 Mbps goodput, which just
  means the links were short and saturated. The honest per-envelope network cost is the
  0.83 s AODV latency.
- **Two malicious updates were aggregated before TA revocation (R17, R52).** That's the
  one security regression against pre-PR runs, which accepted none. Tuning the anomaly
  screen under IID data and a decaying LR is the natural next thing to investigate.
- **v2's small energy and latency numbers come from its tiny MLP and in-process
  transport, not better networking.** The fair accuracy comparison with v1 is the test
  split: 82.4% vs 76.4%.

## Reproducing the run

Run `python main.py --rounds 100 --vehicles 28 --output results/final2` on `master` at
or after commit `260a8b8`, with seed 42. Vehicle sampling, mobility, channel ARQ and
minibatches are all seeded, so reruns match exactly apart from wall-clock timings.

As the README notes, the committed `results/reference_seed42_12rounds/` was produced
before PR #1 changed the defaults (`--alpha 0.5`, no LR decay), so it won't reproduce
exactly. The closest current command is
`python main.py --compare --rounds 12 --alpha 0.5 --lr-decay 1.0`.
