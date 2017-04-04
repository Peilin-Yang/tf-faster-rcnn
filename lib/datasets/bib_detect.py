# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb

class bib_detect(imdb):
    def __init__(self, root): # `root` could be either a folder or a file
        imdb.__init__(self, 'bib_detect')
        self._root = os.path.abspath(root)
        self._classes = ('__background__', # always index 0
                         'bib')
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        # Bib specific config options
        self.config = {'cleanup'     : False,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._root), \
                'Path does not exist: {}'.format(self._root)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def _load_image_set_index(self):
        """
        List all files in the folder
        """

        image_index = []
        print(self._root)
        if os.path.isdir(self._root):
            for root, dirs, files in os.walk(self._root):
                for name in files:
                    image_index.append(os.path.abspath(os.path.join(root, name)))
        elif os.path.isfile(self._root):
            image_index.append(os.path.abspath(self._root))
        else:
            raise NameError('Please provide the valid image paths!! Exit...')

        return image_index
