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
