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

`choir-bench.py` records into the same file under its own harness name. The two
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
which reads on a phone as well as a desktop. Two moving parts, and only because
Tailscale will not serve a *path* without root while it proxies a *port* for
anybody:

    systemctl --user enable --now homr-report     # fixturecheck/homr-report.service
    tailscale serve --bg --https=8124 127.0.0.1:8125

With root it is one command and no service at all, and that is the better shape
if you would rather type a password once:

    sudo tailscale serve --bg --https=8124 --set-path=/ \
        ~/.local/share/homr-fixturecheck/report

The static server binds **loopback only**. The tailnet reaches it through
`tailscale serve`, which authenticates; binding it wider would put the report on
every network this machine is on.

Set `FIXTURECHECK_REPORT_URL` to that address and a run prints the link instead
of a path, and `QUALITY.md` gains one to the pictures.

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
any of them is below 100% exits non-zero. They are small single systems this
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
