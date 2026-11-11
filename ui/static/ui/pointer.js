/* `pagePointer.showCard(key)` shows a figure's card: it opens the tab the card
 * is on, scrolls it into view and outlines it until the person clicks
 * somewhere else. The page's own links (`data-points-at="<key>"`) and the
 * assistant's highlight act (assistant/static/assistant/act.js) both use it,
 * so they find and show a card alike. `outline(el)` is the outline alone, for
 * the assistant's other targets. */
(function () {
    "use strict";

    function showTab(name) {
        var tab = document.querySelector('.fig-tab[data-tab="' + CSS.escape(name) + '"]');
        if (!tab) return false;
        if (tab.getAttribute("aria-selected") !== "true") {
            document.dispatchEvent(new CustomEvent("assistant:show-tab", {detail: {tab: name}}));
        }
        return tab.getAttribute("aria-selected") === "true";
    }

    /* The figure *key*'s card: on the tab showing when it has a place there;
     * else on Overview when it is pinned there; else on its home tab. The tab
     * it is found on is shown. A page without tabs has the card itself. */
    function figureCard(key) {
        var k = CSS.escape(key);
        function shown() {
            return document.querySelector('.fig-panel:not(.is-inactive) .fig-slot[data-key="' + k + '"]');
        }
        var here = shown();
        if (!here) {
            var pinned = document.querySelector('#fig-panel-overview .fig-slot[data-key="' + k + '"]');
            var any = document.querySelector('.fig-slot[data-key="' + k + '"]');
            var tab = pinned ? "overview" : any && any.dataset.home;
            if (tab) showTab(tab);
            here = shown();
        }
        return here || document.querySelector('[data-figure="' + k + '"]');
    }

    /* What is outlined stays outlined until the person clicks somewhere else
     * on the page, or something else is shown. Clicking in the assistant's
     * chat, or on the link that showed it, leaves it. */
    var outlined = null;
    function unoutline() {
        if (outlined) outlined.classList.remove("assistant-highlight");
        outlined = null;
    }
    document.addEventListener("pointerdown", function (e) {
        if (!outlined || outlined.contains(e.target)) return;
        if (e.target.closest && e.target.closest("[data-assistant], [data-points-at]")) return;
        unoutline();
    }, true);

    function outline(el) {
        if (!el) return false;
        unoutline();
        el.scrollIntoView({behavior: "smooth", block: "center"});
        void el.offsetWidth;
        el.classList.add("assistant-highlight");
        outlined = el;
        return true;
    }

    function showCard(key) { return outline(figureCard(key)); }

    /* A link carrying `data-points-at="<key>"` shows that card, and one
     * carrying `data-shows-tab="<tab>"` opens that tab and scrolls to its
     * tabs. Their hrefs stay the way there without script. */
    document.addEventListener("click", function (e) {
        var link = e.target.closest && e.target.closest("[data-points-at], [data-shows-tab]");
        if (!link) return;
        if (link.dataset.pointsAt) {
            if (showCard(link.dataset.pointsAt)) e.preventDefault();
            return;
        }
        var tab = document.getElementById("fig-tab-" + link.dataset.showsTab);
        if (!tab) return;
        showTab(link.dataset.showsTab);
        tab.scrollIntoView({behavior: "smooth", block: "nearest"});
        e.preventDefault();
    });

    window.pagePointer = {showCard: showCard, figureCard: figureCard, showTab: showTab, outline: outline};
})();
