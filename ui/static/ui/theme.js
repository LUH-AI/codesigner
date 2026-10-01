/* Plots drawn to match the page's theme.
 *
 * The server builds every figure with Plotly's own light template, which sets
 * the grey plotting area, the white grid and the dark text. In the dark theme
 * that template is swapped for the one below as each figure is drawn — on the
 * way into `Plotly.newPlot` and `Plotly.react`, which every figure on the page
 * goes through, the page script's and acquisition.js's alike — so no figure
 * needs to know there is a theme at all. Colours the figures choose for
 * themselves (the trials, the prior, the selection) are left as they are.
 *
 * Loaded straight after plotly.min.js, before anything draws. The theme comes
 * from `data-theme` on <html>; "system" is read off the operating system's
 * setting when the page loads.
 */
(function () {
    "use strict";
    if (!window.Plotly) return;

    function dark() {
        var chosen = document.documentElement.getAttribute("data-theme");
        if (chosen === "dark") return true;
        return chosen === "system" && !!window.matchMedia
            && window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    if (!dark()) return;

    var TEXT = "#e6e7eb", PLOT = "#1b1e27", GRID = "#2f3442", LINE = "#4a5063";
    var axis = {gridcolor: GRID, linecolor: LINE, zerolinecolor: LINE,
                tickcolor: LINE, title: {font: {color: TEXT}}};
    var sceneAxis = {backgroundcolor: PLOT, gridcolor: GRID, linecolor: LINE,
                     zerolinecolor: LINE, showbackground: true};
    var TEMPLATE = {
        layout: {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: PLOT,
            font: {color: TEXT},
            colorway: ["#8a92ff", "#ff7a5c", "#2ed3a2", "#c58cff", "#ffb35c",
                       "#4fd8f0", "#ff7aa8", "#c3e77a", "#ff9ad5", "#ffd25c"],
            xaxis: axis, yaxis: axis,
            scene: {xaxis: sceneAxis, yaxis: sceneAxis, zaxis: sceneAxis},
            legend: {bgcolor: "rgba(0,0,0,0)", font: {color: TEXT}},
            hoverlabel: {bgcolor: "#262a36", bordercolor: LINE, font: {color: TEXT}},
            modebar: {bgcolor: "rgba(0,0,0,0)", color: "#8b8f9c", activecolor: TEXT},
            coloraxis: {colorbar: {outlinecolor: LINE, tickcolor: LINE}},
            annotationdefaults: {font: {color: TEXT}},
        },
        data: {
            parcoords: [{labelfont: {color: TEXT}, tickfont: {color: TEXT},
                         rangefont: {color: TEXT}}],
            heatmap: [{colorbar: {outlinecolor: LINE, tickcolor: LINE}}],
            scatter: [{marker: {colorbar: {outlinecolor: LINE, tickcolor: LINE}}}],
            pie: [{outsidetextfont: {color: TEXT}}],
        },
    };

    function themed(layout) {
        var out = {}, key;
        for (key in layout || {}) {
            if (Object.prototype.hasOwnProperty.call(layout, key)) out[key] = layout[key];
        }
        out.template = TEMPLATE;
        return out;
    }

    ["newPlot", "react"].forEach(function (name) {
        var original = window.Plotly[name];
        window.Plotly[name] = function (gd, data, layout, config) {
            /* The one-argument form passes a whole figure. */
            if (data && !Array.isArray(data) && data.data) {
                return original.call(this, gd, Object.assign({}, data, {layout: themed(data.layout)}));
            }
            return original.call(this, gd, data, themed(layout), config);
        };
    });
})();
