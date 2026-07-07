export const meta = {
  name: 'enrich-briefs',
  description: 'Author plain-language practitioner briefs for the experiment ids passed as args',
  phases: [{ title: 'Author', detail: 'one agent per experiment writes the plain-language brief' }],
}

let _ids = args;
if (typeof _ids === 'string') { try { _ids = JSON.parse(_ids); } catch (e) { _ids = _ids.split(/[\s,]+/).filter(Boolean); } }
const EXP_IDS = Array.isArray(_ids) ? _ids : (_ids && _ids.ids) || [];
const REPO = '/home/ericflo/Development/small-model-experimentation';

const BRIEF_SCHEMA = {
  type: 'object',
  required: ['verdict_tag','verdict_tone','concept_primer','plain_question','plain_answer','why_it_matters','key_numbers','charts'],
  properties: {
    verdict_tag: { type: 'string', description: '<=7 words, declarative fragment, no period, no numbers/acronyms/claim-IDs' },
    verdict_tone: { enum: ['positive','negative','mixed','neutral'] },
    concept_primer: { type: 'string', description: '1-2 sentences ~35 words: the ONE idea needed, via concrete analogy, zero jargon' },
    plain_question: { type: 'string', description: '1 sentence ~25 words ending in ?, everyday terms' },
    plain_answer: { type: 'string', description: '1-3 sentences ~45 words, first word carries the verdict (No./Yes,/It depends), percentages not decimals' },
    why_it_matters: { type: 'string', description: '1-2 sentences ~35 words, actionable practitioner guidance, no numbers/claim-IDs' },
    key_numbers: { type: 'array', maxItems: 4, items: { type: 'object', required: ['label','value','sub'],
      properties: { label: { type: 'string' }, value: { type: 'string' }, sub: { type: 'string' } } } },
    charts: { type: 'array', description: 'one entry per chart in this experiment (match count + index)', items: {
      type: 'object', required: ['index','chart_plain_title','chart_read','chart_takeaway'],
      properties: { index: { type: 'integer' }, chart_plain_title: { type: 'string', description: '<=14 words, plain' },
        chart_read: { type: 'string', description: '1-2 sentences ~35 words: name each axis/group/color/reference-line in task terms, say which way is better' },
        chart_takeaway: { type: 'string', description: '1 sentence ~25 words: the concrete visual pattern + its meaning' } } } },
  },
}

const GUIDANCE = `You are authoring the plain-language PRACTITIONER BRIEF for ONE experiment, shown ABOVE THE FOLD on its page (all the jargon-precise detail stays below, untouched). Read ONLY these to author it — never invent facts or numbers:
- ${REPO}/experiments/<ID>/README.md and ${REPO}/experiments/<ID>/reports/report.md (and other reports/*.md if present)
- its verified chart specs: open ${REPO}/knowledge/experiment_viz.json and find the entry keyed "<ID>" (the "charts" array). Author one charts[] entry per chart, with a matching "index" (0-based, in order). If the experiment has no charts there, return "charts": [].

HARD RULES:
(1) Never change a plotted or reported number; every figure you cite must match the chart spec / report exactly.
(2) Translate for an outsider: decimals->percentages (0.275 -> "27%"); "chance 0.031"/"1 in 32" -> "random-guess level"; a non-significant creep -> "not a real change" (never imply significance the report denies).
(3) BAN LIST — never appear in ANY brief field: claim IDs (C25, C9...), "RANK"/"top-1"/"likelihood ranking"/"channel-matched"/"parse-immune"/"pass@k"/"greedy@1"/"AUROC"/"Wilson CI"/"held-out"/"oracle"/"QLoRA"/"SFT"/"DPO"/"MBPP"/"DSL"/"op"/budget-token notation/"banking"/research-program names/bare metric names. Those live only in the untouched technical text below the fold.
(4) verdict_tone + verdict_tag reflect the PRE-REGISTERED HEADLINE verdict, not the most flattering sub-result: positive=a lever worked; negative=the headline hypothesis was refuted/null; mixed=it depends/partial; neutral=pure measurement/exploratory.
(5) concept_primer introduces exactly ONE idea (prefer analogy); everything downstream reuses only terms it defines.
(6) plain_answer and chart_takeaway lead with the answer/pattern, then the contrast; state the win against the null it beat.
(7) chart_read must let someone who ignores every number still understand what is plotted (name each axis, group, color, reference line in TASK terms; say which way is "better"); chart_takeaway names a concrete visual feature and its one-clause meaning.
(8) Respect the word caps in the schema — brevity is plainness. For a null/negative result stay factual and neutral, not alarmed.
(9) key_numbers: 2-4 tiles, prefer a before->after delta ("27% -> 60%") or "≈ chance"; label = the axis in plain words (never a metric name); value = a human quantity; sub = what it is in plain words. Every number must match the chart data.
Return ONLY the structured object.`

phase('Author')
const results = await parallel(EXP_IDS.map(id => () =>
  agent(GUIDANCE.replace(/<ID>/g, id) + `\n\nEXPERIMENT: ${id}`, { label: `brief:${id}`, phase: 'Author', schema: BRIEF_SCHEMA, effort: 'medium' })
    .then(b => ({ id, brief: b }))
    .catch(() => ({ id, brief: null }))
))
const experiments = {}
let ok = 0
for (const r of results.filter(Boolean)) {
  if (r.brief) { experiments[r.id] = r.brief; ok++ }
}
log(`authored ${ok}/${EXP_IDS.length} briefs`)
return { experiments }
