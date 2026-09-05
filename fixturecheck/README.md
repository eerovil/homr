# fixturecheck

One printed system is a **case**: a picture of it, a reference saying what it
holds, and a parse to be judged against them. The five committed fixtures and
every printed system of every song on this host are the same object, so anything
that can judge a fixture can judge the whole repertoire.

    python -m fixturecheck one laulun-aika-s2     ~20s, or 2s cached
    python -m fixturecheck ten                    ~90s
    python -m fixturecheck all                    ~35 min

**Every run writes `check-report/index.html`.** There is no mode that reports
only numbers: a count can say a system agrees on staves, bars and noteheads and
cannot say whether the parse is the music. Each case page opens with the same
three pictures in the same order — the printed band, homr's output engraved, the
reference engraved — then every note, the page against the output.

The index sorts worst first and marks what moved since the last run of the same
case, which is the question a change actually asks.

## What is compared, and what is not

**homr's output**, not its detection. They are different layers and disagree in
both directions: a note the detector misses still reaches the output, in the
wrong voice, having no stem to place it; a head detected a step off the page is
still written at the right pitch, because pitch is read from the image by the
transformer rather than taken from that geometry. Three reports were filed in one
day claiming homr had misread music when what had been compared was the detector,
so the two are separate here and the page names which is which.

Notes are matched by **when they sound** — the measure walked with the duration
cursor — and not by document order: the reference writes one voice out and backs
up for the next while homr interleaves them, so document order pairs a tenor with
a bass. They are compared on **staff position**, so a male-choir score written an
octave above where it sounds is not counted wrong, and on **which of the staff's
voices** a note is in rather than the voice's number, which neither file agrees
about. A **unison** — one printed head serving two voices, which older choral
engraving does constantly — counts once.

## When the two disagree about the staves

Every note is matched on its staff, so if the reference and the parse hold a
different number of staves, nothing below the first one that diverges is being
compared on the same music. That is **one** wrong answer, counted once as
`structure` -- not the twenty-six lost noteheads it looks like.

Whose wrong answer it is, this cannot work out on its own: there are two files
and no third party. On this repertoire the reference has been at fault as often
as homr, and three systems were written up as homr losing a staff when the page
agreed with homr every time. So the answer is written down instead.
`printed.json` holds how many staves a person counted off the printed band, with
what they saw, and the report then names the side at fault rather than reporting
a difference. Absent means nobody has looked, which is most cases and is not the
same as nobody being wrong.

Adding one is a look at the crop the case page already shows, and a line of JSON.

## A fault opens out into its bar

A fault row says a fault happened. It cannot say what happened: *"homr has this
bar's notes at other beats"* is exactly true and does not say **which** beats,
and *"2 noteheads against 1"* does not say which one survived. Judging whether a
reading is homr's mistake or the reference's meant opening the score, which is
the work the report exists to save.

So every fault row folds out — collapsed, so nothing moves for the rows that
agree — into that bar and staff in full:

- **The notes, both sides.** Every notehead of the bar with its beat, staff
  position and voice, the page beside homr. On `hanget-soi`'s bar 3 this settles
  the question at a glance: the same six notes at the same positions, at beats
  `0 0 0.75 1 1 1.5` on the page and `0 0 1 1.25 1.25 1.5` in homr's reading. A
  duration read differently, and nothing lost.
- **The same bar as pictures**: the printed crop, and that bar in homr's
  engraving and in the reference's. The printed one is the one that matters,
  because on this repertoire it has settled every disagreement that anybody
  checked.

**The engraved pair are cut out of the score's own render, not drawn again.**
Engraving a single bar on its own gives it a title, a fresh layout, and a clef
and key it does not carry in context, so the detail looked like different music
from the system at the top of the page. MuseScore will simply say where each bar
landed:

    MuseScore3 -r 220 score.musicxml -o score.mpos    # a box per measure

Those are absolute page units with the origin at the page corner, and
`pixels = units * dpi / MPOS_UNITS_PER_INCH` converts them. The constant is
measured rather than derived — two scores at 220 and 150 dpi, the crop landing
on the barlines at both edges each time — and it is a rate per inch, so it does
not assume the paper size. The page is rendered **untrimmed**, because `-T`
crops to the ink and moves the origin the boxes are measured from. It is also
fewer MuseScore calls: one render and one `.mpos` per side per case, rather than
an engraving per fault.

**All three are written to one scale**, so that a staff is `STAFF_PIXELS` tall
in every one of them — measured from homr's detection for the scan, and from
MuseScore's own staff size (4 spatia, 1.764 mm each) for the engravings. Nothing
downstream is allowed to resize them again: they are shown at exactly the size
they were written, and the row scrolls if it does not fit. Stretching each to
fill the column it sat in was the bug — it put the same bar on screen at three
different scales, one of them blown up several times.

A measure box covers the whole system, so those two show the bar across every
staff, while the printed crop shows the staff the fault is on. The labels say
so. Cropping the printed one to the system too was tried and is worse on real
scans: choral staves are spaced apart to leave room for the words, so the crop
came out three times the height of the engravings beside it, half of it white
paper.

Cutting a bar out of a MusicXML file is reliable, and so is asking MuseScore
where it drew one. Finding a bar in a photograph is not: it needs the barlines,
which come from **homr's own detection**
(`bars.geometry`, the same segmentation pass `scripts/homr_staves.py` uses in
the choir app, cached per case). The lines have to cut the system into exactly
the bars the score says it holds, and two things are corrected first — each on
its own evidence, never on the count:

- **The opening rule is normally not detected**, being one of a pair with the
  bracket. Whether it is missing is decided by *where it would be*: if the first
  detected line is nowhere near the staff's own left edge, the edge is added
  back as the first boundary.
- **Two lines too close together are one boundary** — a double barline, thin
  against thick, or a line the detector invented beside a real one. Measured
  against the system's own median gap, since how wide a bar is depends on how
  many the system holds. The right-hand line is kept, because at a thin-thick
  double bar the music ends at the thick one.

Anything left over is **refused**, and the row says what was found. A crop of
the wrong bar is a confident picture of the wrong music underneath a finding,
which is the mistake this whole harness exists to stop.

Deciding either of these **by the count is not safe**, and `sammon-ryosto` is
why. Its detection missed the opening rule *and* found one stray line, which
came to exactly four bars' worth of boundaries and was accepted — so every crop
on that case was one bar to the right of the row that named it. Counting cannot
tell two cancelling errors from none.

Song systems keep the first-three-faults treatment in the series. This is the
report, and the report has always been where the music is shown.

## Every run is kept

`check-report/results.json` held the last run and only the last run, so a
three-case run overwrote a sweep of ninety-eight and there was no way to ask
whether any of this is getting better over months. Runs now append to
**`fixturecheck/series.jsonl`**, which is committed — one run per line, so a run
adds a line and rewrites nothing above it.

    python -m fixturecheck status      # instant: the last run of each harness

Every run also rewrites **`QUALITY.md`**, which is the answer to "how good is it
now" for somebody who is not going to run anything: GitHub renders it on a phone
and on a desktop, and it names the homr it is describing. That last part was
half the problem — "the current homr" meant the fork's tip to one reader and the
venv the choir sings from to another, and no number anywhere distinguished them.

`choir-bench.py` records into the same file under its own harness name, keyed by
**the engine it measured** — the install found the way the app finds it
(`HOMR_BIN`, else the venv the installer writes), or a worktree's **last commit
touching `homr/`** rather than its `HEAD`, since a harness commit is not a new
homr. The two
are **never averaged**: this one scores notes across the printed systems of the
repertoire, that one scores staves and bars across the benchmark pages, and one
figure over both would mean nothing.

## Reaching the report

The report is written to **one fixed directory outside every checkout** —
`~/.local/share/homr-fixturecheck/report` by default, `FIXTURECHECK_REPORT` to
move it. It used to be `check-report/` beside the source, so every worktree had
its own and none of them was at an address; reaching one meant ssh and knowing
which branch had produced it, which is most of what made "how good is it now"
unanswerable.

On this host it is on the tailnet at **https://bazzite.taile8d16e.ts.net:8124/**,
which reads on a phone as well as a desktop:

    systemctl --user enable --now homr-report     # fixturecheck/homr-report.service
    tailscale serve --bg --https=8124 127.0.0.1:8125

The server binds **loopback only**. The tailnet reaches it through `tailscale
serve`, which authenticates; binding it wider would put the report on every
network this machine is on. (Tailscale will not serve a *path* without root
while it proxies a *port* for anybody, which is why there is a port here at
all.)

## One folder, more than one history

The report directory is **fixed and shared**, so there is one address to serve.
The series is **per checkout and committed**, so the record travels with the
code that made it. Both are deliberate and neither should move — but together
they leave a seam: two checkouts render into the same folder from two different
records. Each page is right on its own; what is wrong is reading them in
sequence at one URL and taking them for one history. A gate that "went from 5/5
to 3/5" can be somebody rendering from a branch rather than a regression.

So the page **declares its series** the way it already declares its homr, and
the folder remembers what last rendered it (`rendered-from.json`).

**The identity is not the checkout's name**, which the first attempt at this got
wrong: switching branches replaces `series.jsonl` under the same folder name,
and two checkouts on two hosts can share a basename — so a name comparison stays
silent in exactly the cases it exists for. `series_id` is the **resolved path**
and a **root fingerprint** (the first run ever recorded) hashed together: the
path separates two folders that share a name, the root separates two histories
that share a path, and it is stable across appends, which it must be or every
ordinary run would cry wolf.

An id alone still cannot see one thing — the same path and the same first run,
with the history rewritten after the point the last render reached. So that is
asked as a **prefix**: the marker keeps the last run it saw, hashed, and the
next render checks that run is still at that position. `at` is not enough to
compare a position by, since it has second resolution and two runs a moment
apart carry the same stamp.

A render that fails either check says so at the top, naming which of the two it
was. Rendering repeatedly from the same series is the ordinary case and says
nothing, so the banner means something when it appears. The checkout name stays
for reading.

**A worktree therefore needs no setup.** The homr venv is host state, the report
folder is fixed, and a run from anywhere lands at the same address — which is
the whole reason that path moved out of the checkouts. What a worktree does have
is its own branch's series, and that is now visible rather than silent.

## Starting a run from the page

Reading the report needed no ssh; **starting the run that refreshes it still
did** — a checkout, the right `MUSESCORE_CLI_PATH`, and the incantation. So the
page carries a bar: `Run the ten`, `Run everything`, and one named case.
`fixturecheck/serve.py` answers the two routes behind it.

    POST /run    {"tier": "ten"}       queue a run
    GET  /queue  what is running, what is waiting, what finished last

**Presses queue rather than collide.** A run is minutes of every core the
machine has, so a worker takes one at a time and the bar says what is running
and how many are behind it — a button that appears to do nothing for four
minutes is a button people press again. When a run finishes the page reloads
itself, because the numbers below the bar are what changed.

**The command is built from what the harness knows**, never from request text: a
tier has to be one of three, and a case name has to be one `cases.every()`
returns. `Run everything` says it is thirty-five minutes before it starts.

The bar is in every copy of the page and **hides itself when `/queue` does not
answer**, so the same HTML is right whether it is served or opened off disk.

Set `FIXTURECHECK_REPORT_URL` to that address and a run prints the link instead
of a path, and `QUALITY.md` gains one to the pictures.

## The report shows every case, not the last run

The index lists **every case the series knows, each at its own latest
measurement**, with the ones this run re-read shown undated and the rest dated
and greyed. A row last measured under a **different** `(homr, references)` is
shown in italics with the homr that produced it, and is **not in the totals** —
it is history, not evidence about the engine being reported. The page says how
many such rows there are rather than quietly showing a smaller total. Before, it listed only what the run judged — so re-reading a single
system, the cheapest thing you can do, produced a one-case report while the
other ninety-seven pages sat on disk with nothing linking to them.

The gate banner on that page is the same published gate `QUALITY.md` carries,
for the same reason: a banner reporting only this run's share once sat green
above two failing rows.

## What a run keeps, and what it does not

Per case: the counts, and the **first three disagreements** — not the first
three rows, which on an ordinary system are three notes that agree and diagnose
nothing. Three faults are usually enough to tell "homr misread this" from "our
reference is wrong here" months later without re-running anything.

The whole note-by-note table is kept **only for the five committed fixtures**,
and only when it differs from the last one recorded — they are gated at 100%, so
otherwise every run would append the same clean table forever. A run that
matches names the run it matches instead, and a change is then the only table in
the file.

The songs get no table, and that is a licence matter rather than a size one:
this repository is public and their music is not ours.

## The score counts what was lost

`agree / (agree + voice + pitch + size + timing)`. A note homr **lost** and a
beat it **moved** count against it, as well as a note it read wrongly. They did
not before, and under the older denominator a system missing half its notes
could read 100%. **No percentage here is comparable with one quoted before
2026-09-05.**

A case homr could not read at all is not folded in as a zero — it has no notes
to be right or wrong about. It is recorded as `unreadable`, or `unbuildable`
where the case itself could not be made, and counted separately. Before, such a
case was skipped entirely, which made a case that had stopped parsing look
exactly like a case nobody ran.

## The gate

**The five committed fixtures are expected to be perfect**, and a run in which
any of them is below 100% exits non-zero.

**The published gate speaks for all five, whatever a run touched — and for one
homr.** It is built from each fixture's *own latest standing result* **under the
identity being reported**, `(homr, references)`, so a run can only ever move the
fixtures it actually ran, and only for the engine that ran them.

That second half is not a refinement. Aggregating across identities let a 5/5
pass under homr A stay a published pass after a single fixture was re-run under
homr B, with the other four never tested on B at all — a number that does not say
what it describes, arriving inside the fix for a number that did not say what it
described. A fixture not yet judged under the current identity is therefore
`unevaluated` and holds the gate open, exactly like one nobody has ever judged:
"it passed on the old homr" is not a claim about this one. A moved reference key
does the same thing, for the same reason. Reading it off the newest run that
judged anything was wrong twice over: a song-only run published `0/0 passed`,
and — less obviously — `fixturecheck one system4` published `1/1 perfect`, so a
standing `FAIL — 3/5` went green because somebody re-ran a fixture that was
never the problem. A fixture nobody has judged is not a pass either; it is
counted as unevaluated and holds the gate open, because "we have never looked"
and "we looked and it was fine" are different claims.

The command's own exit status stays scoped to what it ran, which is what you
want when re-running one case; the summary is the thing that has to speak for
the whole set. They are small single systems this
repository owns outright; if they are wrong, nothing measured on top of them
means much.

They are not all passing today — `hanget-soi` and `sammon-ryosto` are not — and
the gate is hard anyway rather than set to whatever they currently score.

The eighty-eight song systems are **not** gated. Their references are derived
from cleaned scores that are themselves sometimes wrong, and gating on those
would be gating on our own transcription.

## What is committed

The references are frozen as a **fingerprint per case** (`references.json`), not
as files. A reference is a song's cleaned score imploded back to the shape of
the print, and those scores get edited — so a series built against them partly
measures them, and a number can improve because somebody fixed a score. A hash
makes a reference changing a line in a diff that somebody had to commit.

    python -m fixturecheck freeze      # fingerprint what is on this host now

Committing the references themselves would be better and is not available: this
repository is public and the ninety-three systems are Fazer, Sulasol, Breitkopf
and Fennica Gehrman. The cost of the fingerprint is worth saying plainly — a
fresh clone has the five fixtures and no songs, and cannot rebuild the other
eighty-eight to check any figure in the series against them.

A run whose references have moved says so in its own key (`+drift2`), so a run
measured against something other than the manifest cannot be read as if it were.

## Cost

Cached by what each file depends on: the crop and the reference by the PDF,
bounds and cleaned score; the parse by homr's own code, commit plus whatever is
uncommitted. So an ordinary edit re-runs homr and nothing else, and the poppler
and MuseScore work — half the wall clock — is already on disk. Re-running an
unchanged case takes two seconds.

## Configuration

`CHOIR_REPO` is the choir app's checkout, which owns the songs; it needs
`scripts/make_stem_fixture.py`, which currently lives on that repo's working
branch. `MUSESCORE_CLI_PATH` engraves. Without either, the committed fixtures
still run — they carry their own picture and reference because their songs
cannot be committed.

## Not this

The pytest gate is left alone. It answers yes or no, in CI, in seconds, and
should stay that cheap. This answers how much, and shows the music.
