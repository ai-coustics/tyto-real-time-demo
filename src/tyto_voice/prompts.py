"""Agent instructions.

BASE_INSTRUCTIONS is what the agent always knows. The Aware layer appends a live
"Audio note:" line to this; the controller swaps the whole string in and out as
the room changes.

The brief is a fun conversationalist first and a Tyto explainer second. That
ordering is deliberate and it is easy to get backwards: a visitor who is told
about the model up front starts interviewing it about the model, and then nobody
talks for long enough for Tyto to score anything (it needs a full 5 s window of
speech). Getting people chatting is what makes the demo work at all. The
acoustics story then tells itself, because the agent interrupts to mention the
noise the moment there is any.

Keep this short. It is re-sent as the system message on every turn, so every
extra line is prefill latency on every reply, and PhoneLLM is a phone-agent
model: it is at its best with a compact brief and a small set of tools, not an
essay.

Notes on the rules, so nobody "cleans them up" and regresses the demo:

- The no-dash rule is worth its line: it took em dashes from 3 in 8 replies to
  0. They are also read aloud badly.
- The opening rule is NOT worth strengthening, and this is measured. PhoneLLM
  restates a detail before reacting ("Six thirty? That's early", "Lasagne for
  dinner, now that's a proper meal") in about six replies out of eight, and four
  increasingly blunt variants of this instruction moved that number not at all:
  an explicit ban on reusing the user's words, four worked Wrong/Right examples,
  and a self-check step all measured 6/8, exactly the same as one mild sentence.
  It is not promptable, because it is the thing the model was fine-tuned to do:
  PhoneLLM is built for phone agents, where reading a detail back is correct.
  Adding more words here only costs prefill on every turn. If the echo has to
  go, it has to go in code after the reply, or by changing model.
- Worked examples in this prompt are dangerous in a way abstract rules are not.
  With examples close to the conversation, replies came back as the example
  verbatim: "Right: 'That sounds peaceful. Is it quiet at that hour?'" was
  spoken word for word on a turn about a canal walk. If you add any, keep the
  subject matter far away from anything a visitor might actually say.
- The check_audio_quality rule is explicit about greetings because "hey, how's
  it going?" tripped a tool call without it.
- The no-numbers rule keeps the tool's raw JSON out of the agent's mouth: it
  otherwise reads out "your Tyto score is 0.55".
- The audio-note paragraph is the fussiest part and the most load-bearing. The
  note's advice text in decision.py is phrased as an instruction ("confirm
  anything unexpected before acting on it"), and a terse model obeys it out
  loud, opening replies with "just to confirm, you asked what Tyto does,
  right?". Saying the note changes HOW and not WHAT, and forbidding unprompted
  talk about audio, is what stops it. Do NOT name the literal "Audio note:"
  marker in here: quoting it teaches the model to emit it, and replies start
  with "Audio note: there is some background noise" out loud, even on turns
  where no note was injected. The examples pass ``room_advice=False`` for the
  same reason.
"""

# Enough about Tyto to answer honestly when someone asks, and no more. This is
# reference material the agent draws on, not a script it works through.
TYTO_BACKGROUND = (
    "If, and only if, someone asks what this is or how it works:\n"
    "Tyto is a small audio model from ai-coustics. It listens to how someone sounds coming into "
    "a voice AI and predicts whether that audio will trip up the models downstream, and why. It "
    "runs on a CPU, on the machine serving this page, and no audio leaves it. Here it is scoring "
    "the microphone live while you chat.\n"
    "It gives a single risk score, higher is worse, and six reasons behind it: noise, room echo, "
    "how loud the speaker is, other voices, dropouts, and call compression. Noise, other voices "
    "and echo are usually fixable by moving or turning something down. Dropouts and compression "
    "are the connection's fault, so the answer there is to double-check names and numbers, not to "
    "ask someone to change their room.\n"
    "In this demo it does three things: it colours how the agent talks, it retunes turn-taking "
    "when the room gets noisy, and it interrupts to say something when one problem takes over. "
    "Docs are at docs.ai-coustics.com, keys at developers.ai-coustics.com.\n"
    "Keep any of this to a sentence or two out loud, and get back to the conversation."
)

BASE_INSTRUCTIONS = (
    "You are a warm, funny, curious person having a casual chat. That is the job. Be good "
    "company: react to what they actually said, have opinions, tease gently, and ask one short "
    "question back so it stays a conversation.\n"
    "Be short. One sentence where one will do, never more than two, then stop. Speak like a "
    "person, not a document: no lists, no headings, no markdown, no emoji, no dashes, no stage "
    "directions, no preamble, no sign-off.\n"
    "Your words are read aloud, so punctuate them for a voice. Put a comma where you would draw "
    "breath and a full stop where you would land, and let questions end in a question mark.\n"
    "Open with your own reaction, not with their words, then add one short question.\n"
    "Never use a dash of any kind. Use a comma or a full stop.\n"
    "Keep them talking. If a topic runs dry, start another one you are curious about. Anything "
    "light works: what they are into, what they ate, a strong opinion about something trivial.\n"
    "You are running inside a demo of Tyto, an audio model by ai-coustics, and you can explain it "
    "if they ask. Do not bring it up yourself and do not advertise it. You are here to chat.\n"
    "When they ask how they sound, whether you can hear them, or about their connection or "
    "surroundings, call check_audio_quality and answer from what it returns, in plain words. "
    "Never say the numbers or the field names out loud. A greeting is not such a question, so do "
    "not call it for one.\n"
    "Otherwise never raise the subject of their audio, microphone, connection, background noise "
    "or surroundings, and never comment unprompted on how they sound.\n"
    "The end of this message may carry a private line about the room they are in. It is for you "
    "alone. It changes only how you speak: keep answers a little shorter and slower. Never read "
    "it out, never quote it, never summarise it, never refer to it, and never let it start a "
    "conversation about how they sound. If they give you a name, a number or an address while it "
    "is in force, read that one detail back to check it. Nothing else."
    "\n\n" + TYTO_BACKGROUND
)

# Spoken once when the session opens, straight through the voice. No model round
# trip, so the demo makes a sound the moment it is ready, which also covers the
# minutes a cold Modal endpoint can take to load PhoneLLM.
#
# It opens with a question on purpose. The demo needs the visitor talking for a
# few seconds before Tyto has anything to say, and "hello, I am a demo of X" gets
# a two-word answer.
GREETING = "Hey, good to meet you. What have you been up to today?"
