# The trials table as a column of its own

## Why

The trials table is the figure you look things up in while reading a chart —
which trial that outlier was, what configuration produced the point you just
clicked. Doing that by scrolling to the bottom of the page and back is the
reason it wants to be *beside* the figures rather than after them.

The earlier attempt at a wide layout tried to make every figure four across.
This is the simpler thing: the figures keep the two-column arrangement they
already have, and the table takes two columns of its own beside them, full
height.

## How

`Figure.in_side_column`, beside `in_sidebar` and the rest. Only `Trials` sets
it. `ui/views.py` splits the shown figures three ways now — the grid, the
column, the sidebar — all in catalog order.

The markup is two panes:

```html
<div class="figure-layout">
    <div class="grid-2x2"> …every other figure… </div>
    <div class="figure-column"> …the trials table… </div>
</div>
```

and one layout decision in CSS:

```css
.figure-layout { display: flex; flex-direction: column; gap: 1.5rem; }
@media (min-width: 1600px) {
    main.content  { max-width: none; }
    .figure-layout { display: grid; grid-template-columns: 1fr 1fr; }
}
```

Stacked by default, which is the layout there has always been: the column's one
figure is last in catalog order and full width, so stacking puts it exactly
where it already was. The layout it leaves is the layout it falls back to, and
nothing in the page script has to know which is showing.

`main.content`'s `max-width: 1300px` lifts with the second pane or the two
halves have nowhere to go.

## Full height, scrolling inside itself

A column that ran past the figures and had to be reached by scrolling the page
would not be beside anything. So in the wide layout the table's section is
sticky at the top of the viewport with a `max-height`, and the table scrolls
within it:

```css
.figure-column > section { position: sticky; top: 1.5rem;
                           max-height: calc(100vh - 3rem); display: flex; }
.figure-column .table-wrap { flex: 1; overflow: auto; }
```

The pager beneath it stays put while the rows scroll, which is what makes fifty
rows in a fixed-height box readable rather than cramped.

## Verify

```
python -m pytest -q -m "not slow"     # 974 passing
python manage.py check
```

A test asserts the two panes exist, that the table is in the column and not in
the grid, and that it is still last in catalog order — which is what makes the
narrow fallback identical to the old layout rather than merely similar.

In the browser, since none of this is exercised by the suite:

- widen past the threshold and confirm the figures keep their two columns while
  the table takes the other half
- scroll the figures and confirm the table stays beside them
- narrow again and confirm it returns to the bottom, full width, exactly as before
- sort, page, and select a trial in both layouts
