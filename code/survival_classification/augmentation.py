

def aug_sample(img, flip=False, translation=False, rotation=False, intensity=False):
    if (flip==True):
        if np.random.rand(1)[0] < 0.9:
            # Flip x
            img = np.flip(img, axis=3);
        if np.random.rand(1)[0] < 0.9:
            # Flip y
            img = np.flip(img, axis=2);
        if np.random.rand(1)[0] < 0.9:
            # Flip z
            img = np.flip(img, axis=1);

    if (translation==True):
        MARGIN = 10;
        ioff = np.random.randint(-MARGIN, MARGIN);
        joff = np.random.randint(-MARGIN, MARGIN);
        koff = np.random.randint(-MARGIN, MARGIN);

    if (rotation==True):
        


