from PIL  import Image


def imgcat(paths, xy, w=64, pad=2):
    ncol, nrow = xy
    assert len(paths) == ncol * nrow

    width = ncol * w + (ncol - 1) * pad
    heigth = nrow * w + (nrow - 1) * pad
    iout = Image.new('RGB', (width, heigth), (255, 255, 255))
    idx = 0
    for i in range(nrow):
        for j in range(ncol):
            img = Image.open(paths[idx]).resize((w, w), Image.ANTIALIAS)
            iout.paste(img, (j * (w + pad), i * (w + pad)))
            idx += 1

    return iout
