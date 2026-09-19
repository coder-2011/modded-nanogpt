#pragma once

struct TailReferenceView {
    int tokens;
    const void *x, *sources[10], *w1, *w2, *bias, *wg, *dy;
    const void *output, *dx, *dsources[10], *dw1, *dw2, *dbias, *dwg;
    const void *mu, *mug, *mixed;
};

void check_tail_aten(const TailReferenceView &view);
