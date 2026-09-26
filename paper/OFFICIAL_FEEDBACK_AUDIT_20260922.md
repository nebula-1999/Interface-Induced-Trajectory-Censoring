# Follow-up to supplied OpenReview LLM feedback

No GPU run or new inference. This audit corrects measurement descriptions, not
historical trajectories. Existing unrelated writeup changes are preserved.

## Trace-to-source checks

Run `python3 analysis/review_counts_audit.py` against the retained local archive.

- Mean turns 0.96 is the recorded code-tested-step counter, not assistant turns.
  Four Hermes outputs contain no extractable code; all 100 have an assistant record.
  Renamed the table field without fabricating replacement experimental numbers.
  Adapter mean assistant records is 1.90, tested-step mean 1.82. Historical ReAct
  v3 gives 1.92 tested steps; unified v14 gives 1.93. Both have 40 items reaching
  two tested steps but 41 reaching two assistant records. Consequently the
  conditional-rescue subset is not simply all items receiving feedback.
- Figure 3 previously mixed Llama executed 74 with parsed 98. Both bars now use
  parsed counts: 97 versus 98, verified from first-turn parse_mode.
- 319 HumanEval / 677 MBPP are totals across two channels, not distinct tasks.
  Multi is 164+378=542; repair is 155+299=454. Historical multi excluded two items.
- Both a14 and roleprobe record strength=optional plus sys_file. Archived code
  loads the file after selecting the default prompt. The retained roleprobe file
  explicitly mandates run_tests. The a14 override file is missing locally, so its
  exact original bytes are not independently verified. Inventory now discloses this.
- Full first-turn archive cap hits by Coder size: 0/100, 0/100, 1/100, 0/100,
  18/100. These are stored-length cap hits, not observed omitted continuations.
- Human annotation intervals recomputed with sorted IDs, seed 7, 5000 bootstrap
  samples using validation/score_annotator2.py's functions. A1/A2 intervals:
  n=97 [0.595,0.866], n=62 [0.893,1.000]. Repeated tables now agree.

## Method and presentation changes

- Explicitly discuss training tool-role versus evaluation user-role feedback and
  limited transfer; no claim that native FC learning has been ruled out.
- Explain evaluator extraction: tool_call JSON code, then last code fence/raw text.
  This is not a universal syntax adapter. Budget/capacity/credit assignment remain
  alternative explanations, not measured causes of the negative result.
- Separate constructive generation suppression from masking an emitted action.
- Distinguish required-mode screening from clean-arm admissibility; zero auto
  parsing is not an exclusion criterion.
- Constrained generation is a different possible remedy, not dismissed because it
  was not exposed by the tested rollout path. No latency or universal-success claim.
- Name the 7B-Instruct checkpoint for external experiments; collapse redundant
  scale-table columns; clarify unparsed indicators and count units.
- No-envelope is a first-failure category of all unparsed outputs, not proof of a
  tool attempt. At 1.5B it includes non-call outputs; payload validity is untested.
- Correct wrong-tool header; remove misleading significance emphasis; give
  conditional rescue percentages as 42.5% and 24.3%; clarify cross-table denominators.
- Bind small affected table captions to their bodies; replace unsupported indicator
  glyph; keep raw archive/code paths unchanged so reproduction instructions resolve.

## Remaining evidence boundaries

### Figure 1 metric alignment (final review)

- Replaced the mislabeled `Observation` bar with `>=2 tested steps` in the
  generator, plotted title, and both paper captions. The generator now uses
  `n_turns >= 2`, not `len(turns) >= 2`: an unparsed assistant reply can add a
  record without adding a code-bearing test. Recomputed counts are 0, 0, 37,
  and 40 across the four arms; ReAct's old 41 counted assistant records.
- The execution-to-second-test decline is not evidence of feedback-return
  failures. No observation-return count was inferred from these records.
- Changed the introduction to "we detect no improvement under" the shared
  evaluation; restricted the single-tool limitation to controlled scale and
  RL experiments, not BFCL or tau-bench.
- Rebuilt both PDFs and visually checked Figure 1 in each; no new experiments.

Final validation: anonymous body remains eight pages (references begin on page 9),
32 pages including references/appendices; public long version is 43 pages.
Both compile without overfull boxes or unresolved references. Relevant figure/table
pages were rendered and inspected. Unpacked supplement verifier passed on 11,952
outcomes and event records; recomputed statistics match the packaged JSON exactly.
No upload or public push was performed.

- No native-FC checkpoint evaluation, second runtime, new seed or AppWorld run added.
- Historical a14 prompt bytes remain unavailable; this limitation is disclosed,
  not repaired by guessing from its filename.
- Source files retained in the research archive are not all included in the
  anonymous supplement; its verification-scope inventory remains authoritative.
- Neither p=0.25 nor unchanged rescue totals proves equivalence or absence of learning.
