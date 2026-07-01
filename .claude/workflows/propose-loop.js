export const meta = {
  name: 'propose-loop',
  description: 'Proposer↔critic refinement loop. kaggle-proposer drafts N experiment proposals from graph.md + journal.md + memory; kaggle-proposal-reviewer accepts/revises/drops each; the proposer revises on the feedback; loop until the critic is happy or maxIters is reached. Returns the refined proposals — the orchestrator then registers them and hands EVERY one to kaggle-developer. No human, no file writes in the script; the agents read state under comps/<slug>/.',
  phases: [
    { title: 'Propose',  detail: 'kaggle-proposer drafts N proposals from graph.md + journal.md + memory' },
    { title: 'Critique', detail: 'kaggle-proposal-reviewer accepts/revises/drops each proposal' },
    { title: 'Revise',   detail: 'kaggle-proposer applies the feedback' },
  ],
}

// ---------------------------------------------------------------------------
// args: { slug, nProposals=3, maxIters=2 }
//   slug        — competition slug (comps/<slug>/ already validated + baselined)
//   nProposals  — proposals the proposer drafts (default 3)
//   maxIters    — max critique→revise cycles before returning (default 2)
// Returns { proposals:[{slot, op, parents, parent_src, family, desc, uses_data, change, context, hypothesis, target}], all_good, note }.
// ---------------------------------------------------------------------------
const SLUG = (args && args.slug) || 'playground-series-s6e6'
const N = Math.max(1, (args && args.nProposals) || 5)
const MAX_ITERS = Math.max(0, (args && args.maxIters) ?? 2)
const ROOT = `comps/${SLUG}`
if (!SLUG) return { error: 'args.slug is required (the competition slug under comps/<slug>/)' }

const STANDING = `Standing contract: the repo-root CLAUDE.md (graph/DAG semantics, search policy,
leakage discipline, dates from \`date -u\` UTC, every script via \`uv run\`). Spec: ${ROOT}/spec.md.
Graph: ${ROOT}/graph.md. Journal: ${ROOT}/journal.md. Cross-comp lessons: MEMORY.md.`

// ===== schemas =============================================================

const PROPOSAL = {
  type: 'object', additionalProperties: false,
  required: ['slot', 'op', 'parents', 'parent_src', 'family', 'desc', 'uses_data', 'change', 'context', 'hypothesis', 'target', 'well'],
  properties: {
    slot: { type: 'integer', description: 'stable 1-based handle for this proposal' },
    op: { type: 'string', enum: ['draft', 'improve', 'debug', 'combine'] },
    well: { type: 'string', enum: ['exploit', 'data', 'outside', 'wildcard'], description: 'the idea well this proposal came from (see kaggle-proposer.md) — a round must not be 100% exploit' },
    parents: { type: 'array', items: { type: 'string' }, description: '["root"] for a draft; the 1 node for improve/debug; the 2+ merged nodes for combine' },
    parent_src: { type: 'string', description: 'repo-relative dir to copy from, e.g. comps/<slug>/champion/src or comps/<slug>/nodes/node_0006/src' },
    family: { type: 'string', description: 'gbdt|nn|linear|darts|ensemble|baseline' },
    desc: { type: 'string', description: '≤8-word label (Mermaid + table row)' },
    uses_data: { type: 'array', items: { type: 'string' }, description: 'engineered feature-sets consumed (data.md fs_ ids); [] = base only. Name any NEW set fs_<name> in `change`.' },
    change: { type: 'string', description: 'the ONE atomic change, 2–4 lines' },
    context: { type: 'string', description: 'FREE-FORM build context — the developer\'s spec: the concrete HOW of the experiment + every reference worth READING (parent src dir, data.md recipe, refs/ kernel, discussions.md/MEMORY.md line). Never prescribes which files/functions to write.' },
    hypothesis: { type: 'string' },
    target: { type: 'string', description: 'metric + direction; beats parent if CV better than <parent cv>' },
  },
}

const PROPOSALS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['proposals', 'frontier'],
  properties: {
    proposals: { type: 'array', items: PROPOSAL },
    frontier: { type: 'string', description: 'one line: where the search stands' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['reviews', 'all_good', 'note'],
  properties: {
    reviews: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['slot', 'verdict', 'feedback'],
        properties: {
          slot: { type: 'integer' },
          verdict: { type: 'string', enum: ['accept', 'revise', 'drop'] },
          feedback: { type: 'string', description: 'one concrete line — what to change' },
        },
      },
    },
    all_good: { type: 'boolean', description: 'true only if every proposal is accept' },
    note: { type: 'string', description: 'one-line overall take' },
  },
}

// ===== prompts =============================================================

const proposePrompt = () => `${STANDING}

JOB: PROPOSE. Competition "${SLUG}". Draft ${N} INDEPENDENT experiment proposals under the search policy
(distinct parents/families so they can build in parallel). Read ${ROOT}/graph.md, ${ROOT}/data.md (reuse an
existing feature-set before re-engineering one), ${ROOT}/journal.md tail, the node records you cite,
${ROOT}/spec.md, and the relevant MEMORY.md lines first.
Each proposal: slot, op, parents, parent_src (dir to copy), family, desc (≤8 words), uses_data (data.md fs_ ids
it consumes; [] = base only; name any NEW set fs_<name> in change with its leak-safety class), change (the ONE
atomic change), context (free-form — the developer's spec: the concrete HOW + every reference worth READING;
never which files to write), hypothesis, target, well (exploit|data|outside|wildcard — tag the idea well per
kaggle-proposer.md; never 100% exploit). Write NOTHING. Return ONLY the structured object.`

const critiquePrompt = (proposals) => `${STANDING}

JOB: critique these experiment proposals for "${SLUG}" (you are the kaggle-proposal-reviewer).
Proposals: ${JSON.stringify(proposals.proposals)}
For each: verdict accept|revise|drop + one concrete line of feedback. Read ${ROOT}/graph.md + ${ROOT}/journal.md
to catch anything already tried or redundant. Set all_good=true only if EVERY proposal is accept.
Return ONLY the structured object.`

const revisePrompt = (proposals, review) => `${STANDING}

JOB: REVISE. Competition "${SLUG}". Apply this feedback to the proposals and return the updated set
(drop/replace/sharpen as told; keep accepted ones unchanged). Write NOTHING.
Proposals: ${JSON.stringify(proposals.proposals)}
Feedback:  ${JSON.stringify(review.reviews)} | ${review.note}
Return ONLY the structured object (same shape as PROPOSE).`

// ===== loop ================================================================

phase('Propose')
let proposals = await agent(proposePrompt(),
  { label: 'propose', phase: 'Propose', schema: PROPOSALS_SCHEMA, agentType: 'kaggle-proposer' })
if (!proposals || !proposals.proposals || proposals.proposals.length === 0) {
  return { proposals: [], all_good: false, note: 'proposer returned no proposals' }
}

let review = null
for (let i = 0; i < MAX_ITERS; i++) {
  phase('Critique')
  review = await agent(critiquePrompt(proposals),
    { label: `critique:${i + 1}`, phase: 'Critique', schema: REVIEW_SCHEMA, agentType: 'kaggle-proposal-reviewer' })
  if (!review || review.all_good) break
  phase('Revise')
  const revised = await agent(revisePrompt(proposals, review),
    { label: `revise:${i + 1}`, phase: 'Revise', schema: PROPOSALS_SCHEMA, agentType: 'kaggle-proposer' })
  if (revised && revised.proposals && revised.proposals.length) proposals = revised
}

const allGood = !!(review && review.all_good)
log(`refined ${proposals.proposals.length} proposals — ${allGood ? 'critic happy' : `iteration cap (${MAX_ITERS}) reached`}`)
return { proposals: proposals.proposals, all_good: allGood, note: review ? review.note : 'no critique run' }
