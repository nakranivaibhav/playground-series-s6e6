| ts (UTC)            | node      | cv     | lb      |
|---------------------|-----------|--------|---------|
| 2026-06-05T11:28Z | node_0000 | 0.333333 | 0.33333 |
| 2026-06-05T11:44Z | node_0001 | 0.964569 | 0.96612 |
| 2026-06-05T14:23Z | node_0007 | 0.965530 | 0.96702 |
| 2026-06-05T15:43Z | node_0010 | 0.965889 | 0.96704 |
| 2026-06-06T13:49Z | node_0017 | 0.966084 | 0.96702 |
| 2026-06-06T14:18Z | node_0020 | 0.966627 | 0.96722 |
| 2026-06-07T12:52Z | node_0029 | 0.969205 | 0.96993 | 10-base stack + RealMLP-ref breakthrough |
| 2026-06-07T16:40Z | node_0041 | 0.969808 | 0.97043 | CORE+CatBoost 15-base stack |
| 2026-06-09T06:43Z | node_0047 | 0.970881 | 0.96242 | CV MIRAGE — specialist tanked LB −0.0080 vs n41; reverted |
| 2026-06-09T11:38Z | node_0055-restack | 0.969794 | 0.97083 | PROBE: CORE15+DCN; +0.0004 LB vs champ (best LB), CV flat — within public noise |
| 2026-06-09T11:54Z | compound4-restack | 0.969661 | 0.97081 | PROBE2: CORE15+n49/50/51/55; LB does NOT compound (≈n55-only 0.97083), CV lower → rebuild EV low |
| 2026-06-10T11:01Z | A1-bank17 | 0.970153 | 0.97073 — public-bank-17 stack on our folds; BEST CV, honest, no test-fit; LB +0.0003 vs champ, <0.97083 best-LB (within noise); finals slot-1 candidate |
| 2026-06-10T12:32Z | A4-vote | n/a | 0.97123 — clout vote-blend (top7 public + bank17); BEST PUBLIC LB (+0.0004 vs prior best 0.97083); slot-2 only, no private guarantee |
| 2026-06-12T13:37Z | node_0067 | 0.969414 | 0.96998 | PROBE: distillation student solo; LB +0.00057 over CV (normal gap); below champ 0.97073, A4 0.97123 — no promote |
| 2026-06-12T17:55Z | node_0070 | 0.970211 | 0.97087 | PROBE: bank17+FT-Transformer; LB +0.00014 vs champ — CV AND LB BOTH up, best honest pick |
| 2026-06-12T17:55Z | node_0069-argmax | 0.970156 | 0.97079 | PROBE: bagged ARGMAX no-thresh; LB +0.00006 vs champ — threshold NOT load-bearing, safer construction wins |
| 2026-06-13T15:16Z | node_0076 | 0.970227 | 0.97073 | honest best stack (FT-T+bag+argmax); LB == champion, BELOW n70 0.97087 — bagging+argmax did NOT compound with FT-T; best CV but n70 still best honest LB |
| 2026-06-13T15:50Z | node_0084-clout | 0.970299 | 0.97036 | CLOUT re-stack +n74(A4-teacher); HIGHEST CV but WORST LB — n74 add is a CV mirage on LB (inflates OOF, tanks public); dead for finals |
| 2026-06-14T10:55Z | node_0091 | 0.970355 | 0.97121 | PROBE→PROMOTE: full-pool L2 stack (C=0.003); BEST honest LB, beats n070 0.97087 by +0.00034; near clout A4 0.97123 but CV-backed |
| 2026-06-17T11:14Z | node_0117 | 0.969272 | 0.97003 | PROBE: finals slot-2 hedge de-risk; LB>CV, no collapse, private-sane |
| 2026-06-18T06:46Z | node_0129 | 0.970410 | 0.97118 | ens-of-ens meta over 6 stacks; user LB probe; CV+0.000056 did NOT translate (LB -0.00003 vs champ) → noise, no promote |
| 2026-06-18T09:18Z | node_0116 | 0.970369 | 0.97110 | finals-contender LB probe; LB LOWEST of the 3 (CV overstated it) → drops from finals pair |
| 2026-06-21T08:38Z | node_0140 | 0.969305 | 0.97009 | PROBE: redshift-error RealMLP solo (fs_zsoft, best solo base), human-directed solo-vs-stack LB diagnostic — ref 53907616 |
| 2026-06-23T12:42Z | node_0140 (15-fold) | 0.969695 | 0.97007 | PROBE — RealMLP n140 recipe at 15-fold (more data/model + 15-model avg); ref 53979854 |
| 2026-06-23T12:42Z | node_0140 (20-fold) | 0.969744 | 0.97021 | PROBE — RealMLP n140 recipe at 20-fold; BEST single-model LB (>0.97); ref 53979857 |
| 2026-06-24T07:02Z | node_0140 (30-fold) | 0.969903 | 0.97010 | PROBE — RealMLP n140 at 30-fold; highest CV but LB BELOW 20-fold (0.97021) → 20%-slice noise, plateau confirmed; ref 54003338 |
