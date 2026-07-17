# Post-closure forensics: the unparseable rows are cap truncations mid-correct-walk

Read from the committed gate generations
(`runs/local/enum_repair_seed88052.jsonl`), 2026-07-16, after the cell
closed on its frozen verdict (nothing here reopens it).

- Of the 21 axis rows without an ANSWER line, 20 end mid-generation
  without terminal punctuation — truncations at the 1,024-token cap —
  and the visible reasoning at the cut shows the model FAITHFULLY
  executing the frozen canonical walk (e.g. "Step 5 has nothing left to
  try in that order. Step 6 is untried. Its action slots are 1 through
  8, in the list order. Slot 1 … is already what step 6 reads. Slot
  2 …"). Eleven of the 21 contain explicit step-change phrasing.
- Reading: the 22.5% canonical-next fidelity UNDERCOUNTS the installed
  discipline — the skill expresses as a verbose linear walk whose token
  cost grows with k and list size, exceeding any bounded budget on
  deep-k rows. This also predicts the family failure independently of
  the frozen verdict: a verbose walker cannot fit repair turns inside a
  bounded episode.
- The successor evidence the frozen closure demanded now exists: the
  failure is EXPRESSION COST, not discipline. The sharper pedagogy is
  "count, don't walk" — the tried list has k entries in canonical
  order, so the target is entry k+1 of the NUMBERED action list: an
  index computation and lookup, constant-cost in k. The training think
  targets already narrate compactly; the model ignored the compactness
  and walked. A count-don't-walk variant trains index arithmetic
  explicitly and grades the think length.
