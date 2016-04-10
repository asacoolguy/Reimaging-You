# Author: Andreas Christian Mueller <t3kcit@gmail.com>
#
# (c) 2012
# Modified by: Paul Nechifor <paul@nechifor.net>
#
# License: MIT

import warnings
from random import Random
import os
import re
import sys
import colorsys
import numpy as np
from operator import itemgetter

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

from query_integral_image import query_integral_image

item1 = itemgetter(1)

# paths for the fonts used. remember to put the bold/cursive ones first
FONT_PATHS = [os.path.join(os.path.dirname(__file__), "fonts/Debby.ttf"),
            os.path.join(os.path.dirname(__file__), "fonts/VarianeScript.ttf"),
            os.path.join(os.path.dirname(__file__), "fonts/MasterOfBreak.ttf"),
            os.path.join(os.path.dirname(__file__), "fonts/BodoniXT.ttf"),
            os.path.join(os.path.dirname(__file__), "fonts/DroidSansMono.ttf"),
            ]
BOLD_FONTS_INDEX = 2

STOPWORDS = set([x.strip() for x in open(os.path.join(os.path.dirname(__file__),
                                                      'stopwords')).read().split('\n')])

class IntegralOccupancyMap(object):
    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        if mask is not None:
            # the order of the cumsum's is important for speed ?!
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1),
                                      axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)

    def sample_position(self, size_x, size_y, random_state):
        return query_integral_image(self.integral, size_x, size_y, random_state)

    def update(self, img_array, pos_x, pos_y):
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:], axis=1),
                                     axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (self.integral[pos_x - 1, pos_y:]
                                     - self.integral[pos_x - 1, pos_y - 1])
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral


def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None,
                      width=None, height=None, ocean=None):
    """Random hue color generation.

    Default coloring method. This just picks a random hue with value 80% and
    lumination 50%.

    Parameters
    ----------
    word, font_size, position, orientation, width, height : ignored.

    random_state : random.Random object or None, (default=None)
        If a random object is given, this is used for generating random numbers.

    """
    if random_state is None:
        random_state = Random()
    return "hsl(%d, 80%%, 50%%)" % random_state.randint(0, 255)


def gradient_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None,
                      width=None, height=None, ocean=None):
    """location based hue color generation.

    Custom coloring based on position of the word. 

    Generates a hue with value 80% and lumination 50%.

    Parameters
    ----------
    word, font_size, orientation, ocean : ignored.

    position, width, height : used to calculate color

    random_state : random.Random object or None, (default=None)
        If a random object is given, this is used for generating random numbers.

    """
    # --- single color based on font frequency ---
    # hue = 240
    # sat = 30
    # lit = 30
    # if font_size > 25 : 
    #     sat += font_size * 30 / 200 + 30
    #     lit += font_size * 20 / 200 + 20


    #sat = 30 + (font_size * 30 / 200)
    #lit = 30 + (font_size * 40 / 200)

    # --- vertical ---
    hue = position[0] * 260 / height
    sat = abs(position[0] - (height / 2)) * 20 / (height / 2) + 60 
    lit = abs(position[0] - (height / 2)) * 20 / (height / 2) + 40 

    # --- horizontal ---
    #hue = position[1] * 260 / width 

    # --- diagonal ---
    #hue = (position[0] + position[1]) * 260 / (width + height)
    #sat = abs(position[0] + position[1] - ((width + height) / 2)) * 20 / ((width + height) / 2) + 70 

    return "hsl(%d, %d%%, %d%%)" % (hue, sat,lit)


# function used to color words based on OCEAN scores
def ocean_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None,
                      width=None, height=None, ocean = None):
    # red 0, green: 137
    # orange red 20, cyan 175
    # orange 36, light blue - 206
    # yellow 58, dark blue 265
    # dark green 108, magenta 297

    if random_state is None:
        random_state = Random()

    # if all 0, then random
    if ocean[0] == 0 and ocean[1] == 0 and ocean[2] == 0 and ocean[3] == 0 and ocean[4] == 0:
        hue = random_state.randint(0, 359)
    else:
        abs_ocean = (abs(ocean[0]), abs(ocean[1]), abs(ocean[2]), abs(ocean[3]), abs(ocean[4]))
        max_value = max(abs_ocean)
        max_index = abs_ocean.index(max_value)
        max_value = ocean[max_index]

        if max_index == 0: # ope. inventive/curious vs. consistent/cautious
            if max_value > 0:
                hue = 58 # yellow
            else:
                hue = 265 # darkblue
        elif max_index == 1: # con. efficient/organized vs. easy-going/careless
            if max_value > 0:
                hue = 297 # magenta
            else:
                hue = 108 # dark green
        elif max_index == 2: # ext. outgoing/energetic vs. solitary/reserved
            if max_value > 0:
                hue = 36 # orange
            else:
                hue = 206 # light blue
        elif max_index == 3: # agr. friendly/compassionate vs. analytical/detached
            if max_value > 0:
                hue = 20 # orange red
            else:
                hue = 175 # cyan
        elif max_index == 4: # neu. sensitive/nervous vs. secure/confident
            if max_value > 0:
                hue = 0 # red
            else:
                hue = 137 # green
    
    sat = abs(position[0] - (height / 2)) * 20 / (height / 2) + 60 
    lit = abs(position[0] - (height / 2)) * 20 / (height / 2) + 40 

    # sat = 30
    # lit = 30
    # if font_size > 25 : 
    #    sat += font_size * 30 / 200 + 30
    #    lit += font_size * 20 / 200 + 20

    return "hsl(%d, %d%%, %d%%)" % (hue, sat,lit)

def get_single_color_func(color):
    """Create a color function which returns a single hue and saturation with.
    different values (HSV). Accepted values are color strings as usable by PIL/Pillow.

    >>> color_func1 = get_single_color_func('deepskyblue')
    >>> color_func2 = get_single_color_func('#00b4d2')
    """
    old_r, old_g, old_b = ImageColor.getrgb(color)
    rgb_max = 255.
    h, s, v = colorsys.rgb_to_hsv(old_r / rgb_max, old_g / rgb_max, old_b / rgb_max)

    def single_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None,
                          width=None, height=None, ocean=None):
        """Random color generation.

        Additional coloring method. It picks a random value with hue and
        saturation based on the color given to the generating function.

        Parameters
        ----------
        word, font_size, position, orientation, width, height : ignored.

        random_state : random.Random object or None, (default=None)
          If a random object is given, this is used for generating random numbers.

        """
        if random_state is None:
            random_state = Random()
        r, g, b = colorsys.hsv_to_rgb(h, s, random_state.uniform(0.2, 1))
        return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(r * rgb_max, g * rgb_max, b * rgb_max)
    return single_color_func


class WordCloud(object):
    """Word cloud object for generating and drawing.

    Parameters
    ----------
    font_path : array of string
        Font paths of the fonts that will be used (OTF or TTF).

    bold_fonts_index (default = BOLD_FONTS_INDEX)

    width : int (default=400)
        Width of the canvas.

    height : int (default=200)
        Height of the canvas.

    prefer_horizontal : float (default=0.90)
        The ratio of times to try horizontal fitting as opposed to vertical.

    mask : nd-array or None (default=None)
        If not None, gives a binary mask on where to draw words. If mask is not
        None, width and height will be ignored and the shape of mask will be
        used instead. All white (#FF or #FFFFFF) entries will be considerd
        "masked out" while other entries will be free to draw on. [This
        changed in the most recent version!]

    scale : float (default=1)
        Scaling between computation and drawing. For large word-cloud images,
        using scale instead of larger canvas size is significantly faster, but
        might lead to a coarser fit for the words.

    min_font_size : int (default=4)
        Smallest font size to use. Will stop when there is no more room in this
        size.

    font_step : int (default=1)
        Step size for the font. font_step > 1 might speed up computation but
        give a worse fit.

    max_words : number (default=200)
        The maximum number of words.

    stopwords : set of strings
        The words that will be eliminated.

    background_color : color value (default="black")
        Background color for the word cloud image.

    max_font_size : int or None (default=None)
        Maximum font size for the largest word. If None, height of the image is
        used.

    mode : string (default="RGB")
        Transparent background will be generated when mode is "RGBA" and
        background_color is None.

    relative_scaling : float (default=0)
        Importance of relative word frequencies for font-size.
        With relative_scaling=0, only word-ranks are considered.
        With relative_scaling=1, a word that is twice as frequent will have twice the size.
        If you want to consider the word frequencies and not only their rank, relative_scaling
        around .5 often looks good.

    upper_font_filter, lower_font_filter: float (default = 0)
        thresholds for font size filtering. all fonts of size that's smaller than the 
        upper_font_filter * max_font_size will be made into sizes smaller than the 
        lower_font_filter * max_font_size

    random_noise: float (default = 0)
        randomly picks out certain words and makes them bigger than usual to inject some noise.
        only happens to words that are larger than 2/3 * largest size
        *** dont use this for now it doesn't quite work ***

    bold_font_threshold (default = number of fonts / 2)
        threshold for which fonts to use for each word depending on word size.
        word with size larger than bold_font_threshold * max_font_size will use bold fonts
        words smaller will use normal fonts


    Attributes
    ----------
    ``words_``: list of tuples (string, float)
        Word tokens with associated frequency.

    ``layout_`` : list of tuples (string, int, (int, int), int, color))
        Encodes the fitted word cloud. Encodes for each word the string, font
        size, position, orientation and color.

    Notes
    -----
    Larger canvases with make the code significantly slower. If you need a large
    word cloud, try a lower canvas size, and set the scale parameter.

    The algorithm might give more weight to the ranking of the words
    than their actual frequencies, depending on the ``max_font_size`` and the
    scaling heuristic.
    """

    def __init__(self, font_path=None, bold_fonts_index=None, width=400, height=200, margin=2,
                 ranks_only=None, prefer_horizontal=0.9, mask=None, scale=1,
                 color_func=gradient_color_func, max_words=200, min_font_size=4,
                 stopwords=None, random_state=None, background_color='black',
                 max_font_size=None, font_step=1, mode="RGB", relative_scaling=0,
                 upper_font_filter=None, lower_font_filter=None, random_noise=0,
                 bold_font_threshold=None):
        if font_path is None:
            font_paths = FONT_PATHS
        else:
            font_paths = [font_path]
        if bold_fonts_index is None:
            self.bold_fonts_index = BOLD_FONTS_INDEX
        else:
            self.bold_fonts_index = bold_fonts_index

        self.font_paths = font_paths
        self.width = width
        self.height = height
        self.margin = margin
        self.prefer_horizontal = prefer_horizontal
        self.mask = mask
        self.scale = scale
        self.color_func = color_func
        self.max_words = max_words
        self.stopwords = stopwords or STOPWORDS
        self.min_font_size = min_font_size
        self.font_step = font_step
        if isinstance(random_state, int):
            random_state = Random(random_state)
        self.random_state = random_state
        self.background_color = background_color
        if max_font_size is None:
            max_font_size = height
        self.max_font_size = max_font_size
        self.mode = mode
        if relative_scaling < 0 or relative_scaling > 1:
            raise ValueError("relative_scaling needs to be between 0 and 1, got %f."
                             % relative_scaling)
        self.relative_scaling = relative_scaling
        self.upper_font_filter = upper_font_filter
        self.lower_font_filter = lower_font_filter
        if bold_font_threshold is None:
            self.bold_font_threshold = 1/2
        else:
            self.bold_font_threshold = bold_font_threshold
        
        self.random_noise = random_noise
        if ranks_only is not None:
            warnings.warn("ranks_only is deprecated and will be removed as"
                          " it had no effect. Look into relative_scaling.", DeprecationWarning)

    def fit_words(self, frequencies):
        """Create a word_cloud from words and frequencies.

        Alias to generate_from_frequencies.

        Parameters
        ----------
        frequencies : array of tuples
            A tuple contains the word and its frequency.

        Returns
        -------
        self
        """
        return self.generate_from_frequencies(frequencies)

    def generate_from_frequencies(self, frequencies):
        """Create a word_cloud from words and frequencies.

        Parameters
        ----------
        frequencies : array of tuples
            A tuple contains the word, its frequency, 
            and a tuple of its ocean scores in OCEAN order

        Returns
        -------
        self

        """
        # make sure frequencies are sorted and normalized
        frequencies = sorted(frequencies, key=item1, reverse=True)
        frequencies = frequencies[:self.max_words]
        # largest entry will be 1 for freq and ocean scores 
        max_frequency = float(frequencies[0][1])
        max_ope = max(frequencies, key=itemgetter(2))[2]
        max_con = max(frequencies, key=itemgetter(3))[3]
        max_ext = max(frequencies, key=itemgetter(4))[4]
        max_agr = max(frequencies, key=itemgetter(5))[5]
        max_neu = max(frequencies, key=itemgetter(6))[6]

        frequencies = [(word, freq / max_frequency, 
                        ope / max_ope, 
                        con / max_con,
                        ext / max_ext, 
                        agr / max_agr, 
                        neu / max_neu) 
                    for word, freq, ope, con, ext, agr, neu in frequencies]

        self.words_ = frequencies

        # variabled used to find actual largest font size
        max_actual_size = 0

        # gives self a random state object for random generation
        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        # checks frequency array size
        if len(frequencies) <= 0:
            print("We need at least 1 word to plot a word cloud, got %d."
                  % len(frequencies))

        # if there is a mask, set the attributes
        if self.mask is not None:
            mask = self.mask
            width = mask.shape[1]
            height = mask.shape[0]
            if mask.dtype.kind == 'f':
                warnings.warn("mask image should be unsigned byte between 0 and"
                              " 255. Got a float array")
            if mask.ndim == 2:
                boolean_mask = mask == 255
            elif mask.ndim == 3:
                # if all channels are white, mask out
                boolean_mask = np.all(mask[:, :, :3] == 255, axis=-1)
            else:
                raise ValueError("Got mask of invalid shape: %s" % str(mask.shape))

            # reset the max font size
            font_size = height * 1 / 4
        else:
            boolean_mask = None
            height, width = self.height, self.width
            font_size = self.max_font_size
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)
        font_sizes, positions, orientations, colors, font_indecies, oceans = [], [], [], [], [], []

        
        last_freq = 1.

        # start drawing grey image
        for word, freq, ope, con, ext, agr, neu in frequencies:
            # select the font size
            rs = self.relative_scaling
            if rs != 0:
                # relative size heuristics. might not want to mess with this?
                font_size = int(round((rs * (freq / float(last_freq)) + (1 - rs)) * font_size))

            # try to find a position
            while True:                
                # set max actual size
                if len(font_sizes) > 0 and len(font_sizes) < 2:
                    max_actual_size = font_sizes[0]

                # font size is in the middle, make it smaller
                # this check will be ignored if max_actual_size is 0 aka not set yet
                # this check will also always fail if upper_font_filter is 0 aka not set
                if font_size < max_actual_size * self.upper_font_filter \
                    and font_size > max_actual_size * self.lower_font_filter:
                    font_size = max_actual_size * self.lower_font_filter

                # randomize a font path to use based on bold_font_threshold
                # if len(font_sizes) is 0 or font_size > max_actual_size * self.bold_font_threshold:
                #     # use bold fonts
                #     font_index = random_state.randint(0, self.bold_fonts_index)
                # else:
                #     # use normal fonts
                #     font_index = random_state.randint(self.bold_fonts_index + 1, len(self.font_paths) - 1)
                font_index = random_state.randint(0, len(self.font_paths) - 1)

                font = ImageFont.truetype(self.font_paths[font_index], font_size)

                # transpose font optionally
                if random_state.random() < self.prefer_horizontal:
                    orientation = None
                else:
                    orientation = Image.ROTATE_90
                transposed_font = ImageFont.TransposedFont(font,
                                                           orientation=orientation)
                # get size of resulting text
                box_size = draw.textsize(word, font=transposed_font)
                # find possible places using integral image:
                result = occupancy.sample_position(box_size[1] + self.margin,
                                                   box_size[0] + self.margin,
                                                   random_state)
                if result is not None or font_size == 0:
                    break
                # if we didn't find a place, make font smaller
                font_size -= self.font_step

            if font_size < self.min_font_size:
                # we were unable to draw any more
                break

            # if len(font_sizes) > 0 and font_size > font_sizes[0] * 2 / 3 and \
            #     random_state.random() < self.random_noise:
            #     font_size *= 2
            # if len(font_sizes) < 5:
            #         font_size *= 2

            x, y = np.array(result) + self.margin // 2
            # actually draw the text
            draw.text((y, x), word, fill="white", font=transposed_font)
            positions.append((x, y))
            orientations.append(orientation)
            font_sizes.append(font_size)
            font_indecies.append(font_index)
            ocean = (ope, con, ext, agr, neu)
            oceans.append(ocean)
            colors.append(self.color_func(word, font_size=font_size,
                                          position=(x, y),
                                          orientation=orientation,
                                          random_state=random_state,
                                          font_path=self.font_paths,
                                          width = width,
                                          height = height,
                                          ocean = ocean))
            # recompute integral image
            if self.mask is None:
                img_array = np.asarray(img_grey)
            else:
                img_array = np.asarray(img_grey) + boolean_mask
            # recompute bottom right
            # the order of the cumsum's is important for speed ?!
            occupancy.update(img_array, x, y)
            last_freq = freq

        print max_actual_size
        self.layout_ = list(zip(frequencies, font_sizes, positions, orientations, font_indecies, oceans, colors))
        return self

    def process_text(self, text):
        """Splits a long text into words, eliminates the stopwords.

        Parameters
        ----------
        text : string
            The text to be processed.

        Returns
        -------
        words : list of tuples (string, float)
            Word tokens with associated frequency.

        Notes
        -----
        There are better ways to do word tokenization, but I don't want to
        include all those things.
        """

        self.stopwords_lower_ = set(map(str.lower, self.stopwords))

        d = {}
        flags = (re.UNICODE if sys.version < '3' and type(text) is unicode
                 else 0)
        for word in re.findall(r"\w[\w']+", text, flags=flags):
            if word.isdigit():
                continue

            word_lower = word.lower()
            if word_lower in self.stopwords_lower_:
                continue

            # Look in lowercase dict.
            try:
                d2 = d[word_lower]
            except KeyError:
                d2 = {}
                d[word_lower] = d2

            # Look in any case dict.
            d2[word] = d2.get(word, 0) + 1

        # merge plurals into the singular count (simple cases only)
        for key in list(d.keys()):
            if key.endswith('s'):
                key_singular = key[:-1]
                if key_singular in d:
                    dict_plural = d[key]
                    dict_singular = d[key_singular]
                    for word, count in dict_plural.items():
                        singular = word[:-1]
                        dict_singular[singular] = dict_singular.get(singular, 0) + count
                    del d[key]

        d3 = {}
        for d2 in d.values():
            # Get the most popular case.
            first = max(d2.items(), key=item1)[0]
            d3[first] = sum(d2.values())

        return d3.items()

    def generate_from_text(self, text):
        """Generate wordcloud from text.

        Calls process_text and generate_from_frequencies.

        Returns
        -------
        self
        """
        words = self.process_text(text)
        self.generate_from_frequencies(words)
        return self

    def generate(self, text):
        """Generate wordcloud from text.

        Alias to generate_from_text.

        Calls process_text and generate_from_frequencies.

        Returns
        -------
        self
        """
        return self.generate_from_text(text)

    def _check_generated(self):
        """Check if ``layout_`` was computed, otherwise raise error."""
        if not hasattr(self, "layout_"):
            raise ValueError("WordCloud has not been calculated, call generate first.")

    def to_image(self):
        self._check_generated()
        if self.mask is not None:
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            height, width = self.height, self.width

        img = Image.new(self.mode, (int(width * self.scale), int(height * self.scale)),
                        self.background_color)
        draw = ImageDraw.Draw(img)
        for (word, count, o, c, e, a, n), font_size, position, orientation, font_index, _, color in self.layout_:
            font = ImageFont.truetype(self.font_paths[font_index], int(font_size * self.scale))
            transposed_font = ImageFont.TransposedFont(font,
                                                       orientation=orientation)
            pos = (int(position[1] * self.scale), int(position[0] * self.scale))
            draw.text(pos, word, fill=color, font=transposed_font)
        return img

    def recolor(self, random_state=None, color_func=None):
        """Recolor existing layout.

        Applying a new coloring is much faster than generating the whole wordcloud.

        Parameters
        ----------
        random_state : RandomState, int, or None, default=None
            If not None, a fixed random state is used. If an int is given, this
            is used as seed for a random.Random state.

        color_func : function or None, default=None
            Function to generate new color from word count, font size, position
            and orientation.  If None, self.color_func is used.

        Returns
        -------
        self
        """
        if isinstance(random_state, int):
            random_state = Random(random_state)
        self._check_generated()

        if self.mask is not None:
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            height, width = self.height, self.width

        if color_func is None:
            color_func = self.color_func
        self.layout_ = [(word_freq, font_size, position, orientation, font_index, ocean,
                         color_func(word=word_freq[0], font_size=font_size,
                                    position=position, orientation=orientation,
                                    random_state=random_state, 
                                    font_path=self.font_paths[font_index],
                                    width = width, height = height,
                                    ocean = ocean),
                         )
                        for word_freq, font_size, position, orientation, font_index, ocean, color in self.layout_]
        return self

    def to_file(self, filename):
        """Export to image file.

        Parameters
        ----------
        filename : string
            Location to write to.

        Returns
        -------
        self
        """

        img = self.to_image()
        img.save(filename)
        return self

    def to_array(self):
        """Convert to numpy array.

        Returns
        -------
        image : nd-array size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return np.array(self.to_image())

    def __array__(self):
        """Convert to numpy array.

        Returns
        -------
        image : nd-array size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return self.to_array()

    def to_html(self):
        raise NotImplementedError("FIXME!!!")
