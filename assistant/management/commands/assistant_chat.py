"""Talk to the assistant from the terminal.

For trying a model and the tools before, or without, the page: each line typed
is a turn, and what the assistant did and said is printed after it.

    manage.py assistant_chat --experiment 12
    manage.py assistant_chat --user ana --experiment 12     (with accounts)

The conversation is kept like any other, and acts as the person named — with
accounts on, as no more than they may. Somebody at this terminal already has
the server, so it is not tied to a browser session.
"""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Talk to the assistant from the terminal."

    def add_arguments(self, parser):
        parser.add_argument("--user", help="The account to act for (with REQUIRE_LOGIN on).")
        parser.add_argument("--experiment", type=int,
                            help="The experiment or draft the conversation is about.")
        parser.add_argument("--conversation", type=int, help="Carry on an earlier conversation.")
        parser.add_argument("--no-stream", action="store_true",
                            help="Ask the model for whole replies rather than streamed ones.")

    def handle(self, *args, **options):
        from assistant.gateway import Context
        from assistant.llm import ChatClient
        from assistant.loop import run_turn
        from assistant.models import AssistantEvent, Conversation
        from assistant.tools.registry import Refused
        from assistant.turns import tools_for
        from ui.permissions import BOTH, VIEW

        owner = None
        if settings.REQUIRE_LOGIN:
            from django.contrib.auth import get_user_model

            if not options["user"]:
                raise CommandError("Accounts are on: name the person to act for with --user.")
            owner = get_user_model().objects.filter(username=options["user"]).first()
            if owner is None:
                raise CommandError(f"There is no account {options['user']!r}.")

        if options["conversation"]:
            conversation = Conversation.objects.filter(pk=options["conversation"],
                                                       owner=owner).first()
            if conversation is None:
                raise CommandError("There is no such conversation for this person.")
        else:
            conversation = Conversation.objects.create(owner=owner)
        if options["experiment"]:
            conversation.experiment_id = options["experiment"]
            try:
                Context(conversation).experiment(options["experiment"], VIEW, BOTH)
            except Refused as refusal:
                raise CommandError(str(refusal))
            conversation.save(update_fields=["experiment"])

        llm = ChatClient.from_settings()
        tools = tools_for(conversation)
        self.stdout.write(f"Conversation {conversation.pk} with {llm.model} at "
                          f"{settings.ASSISTANT_LLM_BASE_URL}. An empty line ends it.")
        seen = conversation.events.order_by("-id").values_list("id", flat=True).first() or 0
        choices = []
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                break
            if not line:
                break
            if choices and line.isdigit() and 1 <= int(line) <= len(choices):
                line = choices[int(line) - 1]
            run_turn(conversation, line, llm=llm, tools=tools, stream=not options["no_stream"])
            choices = []
            for event in AssistantEvent.objects.filter(conversation=conversation, id__gt=seen):
                seen = event.id
                choices = self._show(event) or choices

    def _show(self, event):
        """Print *event*; return the choices it offers, if it offers any."""
        kind, payload = event.kind, event.payload
        if kind == "text":
            self.stdout.write(payload["text"], ending="")
        elif kind == "tool_call":
            self.stdout.write(self.style.NOTICE(f"\n[{payload['tool']} {payload['arguments']}]"))
        elif kind == "tool_result":
            self.stdout.write(self.style.NOTICE(f"[→ {payload['output'][:200]}]"))
        elif kind == "choices":
            self.stdout.write("\n" + payload["question"])
            for number, option in enumerate(payload["options"], start=1):
                self.stdout.write(f"  {number}. {option}")
            return payload["options"]
        elif kind == "turn_end" and payload["reason"] not in ("answered", "waiting"):
            self.stdout.write(self.style.WARNING(
                f"\n[turn ended: {payload['reason']} {payload.get('detail', '')}]"))
        return None
