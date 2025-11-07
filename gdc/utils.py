import ast
import os
import datetime
from types import SimpleNamespace
from cycler import cycler

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import pandas as pd
import unicodedata
import re

from googletrans import Translator


def str2list(s):
    return ast.literal_eval(s)


def str2choices(s):
    return ast.literal_eval(s.replace('null', 'None'))


def get_class_attributes(some_class):
    return [(k, v) for k, v in
            some_class.__dict__.items() if not k.startswith('__')]


def is_in(f):
    if callable(f):
        return f
    elif isinstance(f, (list, set)):
        return lambda x: x in f
    elif isinstance(f, str):
        return lambda x: x in str2list(f)
    else:
        return lambda x: x == f


def is_not_in(f):
    in_f = is_in(f)
    return lambda x: not in_f(x)


def contains(f):
    def contains_this(x):
        if isinstance(x, (list, set)):
            return f in x
        elif isinstance(x, str):
            return f in str2list(x)
        else:
            return f == x
    return contains_this


def contains_all(f):
    def contains_all_this(x):
        if isinstance(x, (list, set)):
            return all([y in x for y in f])
        elif isinstance(x, str):
            return all([y in str2list(x) for y in f])
        else:
            return False
    return contains_all_this


def contains_any(f):
    def contains_any_this(x):
        if isinstance(x, (list, set)):
            return any([y in x for y in f])
        elif isinstance(x, str):
            return any([y in str2list(x) for y in f])
        else:
            return False
    return contains_any_this


def is_less_than(x):
    return lambda y: y < x


def is_weakly_less_than(x):
    return lambda y: y <= x


def is_greater_than(x):
    return lambda y: y > x


def is_weakly_greater_than(x):
    return lambda y: y >= x


def is_between(a, b, inclusive='both'):
    if inclusive == 'both':
        return lambda y: a <= y <= b
    if inclusive == 'left':
        return lambda y: a <= y < b
    if inclusive == 'right':
        return lambda y: a < y <= b


def filter_data(data, dic_conditions):
    this_df = data.copy()
    for key, condition in dic_conditions.items():
        if isinstance(key, tuple):
            this_df = this_df.loc[this_df.loc[:, key].apply(
                is_in(condition), axis=1)]
        else:
            this_df = this_df.loc[this_df.loc[:, key].apply(is_in(condition))]
    return this_df


def str_to_date(this_date):
    if isinstance(this_date, str):
        this_date = datetime.datetime.strptime(this_date, '%Y-%m-%d')
    return this_date


def date_to_str(this_date):
    if isinstance(this_date, datetime.date):
        this_date = this_date.strftime('%Y-%m-%d')
    return this_date


def data_path(data_file, is_folder=False, root_path=None):
    if root_path is None:
        home = os.path.expanduser('~')
        root_path = os.path.join(home, 'Dropbox')
    for root, dirs, files in os.walk(root_path):
        if is_folder:
            for dir_name in dirs:
                if dir_name == data_file:
                    return os.path.join(root, dir_name)
        else:
            for name in files:
                if name.startswith(data_file):
                    return root
    return None


GDC_DATA_PATH = data_path('gdc_data_fr.txt')


def generate_fig_path(path_str):
    def fig_path(filestr):
        return os.path.join(path_str, filestr)

    return fig_path


def normalize_string(s):
    # Convert to lowercase
    # if s is not a string, return s
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # Remove accents
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    # Replace spaces or dashes with underscores
    s = s.replace("'", '_')
    s = s.replace('-', '_')
    s = s.replace(' ', '_')
    # replaec html tags <...> with double underscores
    s = re.sub(r'<[^>]*>', '__', s)
    # Remove non-alphanumeric characters except underscores
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s


class FigSave:
    def __init__(self):
        self.fig_path = data_path(
            'figs_gdc.txt', root_path=GDC_DATA_PATH)

    def __call__(self, fig_name, **kwargs):
        plt.tight_layout()
        # name as .svg if no extension is given
        if fig_name.find('.') == -1:
            fig_name += '.svg'
        plt.savefig(os.path.join(self.fig_path, fig_name), **kwargs)
        plt.show()


def tab2latex(tab, dig=3, caption="Table Caption", label="tab:label",
              pct=False):
    """
    Convert a DataFrame or Series to a LaTeX table with boilerplate.

    Parameters:
        tab (pd.DataFrame or pd.Series): The table to convert.
        dig (int): Number of decimal places to round numeric values.
        caption (str): Caption for the LaTeX table.
        label (str): Label for the LaTeX table for referencing.
        pct (bool): If True, formats numeric values as percentages.

    Returns:
        None
    """
    if isinstance(tab, pd.Series):
        # Convert Series to DataFrame for consistent formatting
        tab = tab.to_frame(name="value")

    # Apply formatting to numeric values
    def format_value(x):
        if isinstance(x, (int, float)):
            if pct:
                return f"{x * 100:.{dig}f}%"  # Convert to percentage
            return f"{x:.{dig}f}"  # Standard formatting
        return str(x)

    tab_formatted = tab.applymap(format_value)

    # Create LaTeX table with boilerplate
    latex_output = f"""
        \\begin{{table}}[h!]
        \\centering
        {tab_formatted.to_latex(index=True, escape=False)}
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\end{{table}}
        """
    print(latex_output)


def millions_formatter(x, pos):
    return f"{int(x / 1e5)/10}"


def percent_formatter(x, pos):
    return f"{x * 100:.0f}%"


def set_formatter(fmtr=millions_formatter, ax=None):
    if ax is None:
        ax = plt.gca()  # Get the current axis if none is provided
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmtr))


def get_pctchange_over_time(df, date_pairs=None, annualized=True):
    data = []
    index = []
    for s, e in date_pairs:
        pct_change = df.loc[[s, e], :].pct_change().iloc[-1]
        if annualized:
            pct_change = (1 + pct_change) ** (1./(e-s)) - 1
        data.append(pct_change)
        index.append('{}-{}'.format(s, e))
    out_df = pd.DataFrame(data=data, index=index)
    return out_df


def setup_translation_for_figures(dest='en'):
    """
    Redefines Matplotlib's text-setting functions
    (xlabel, ylabel, title, tick labels, and legend)
    to automatically translate all text to English before rendering.
    Call this function at the start of your notebook.
    """
    import matplotlib.pyplot as plt
    from googletrans import Translator

    # Initialize the translator
    translator = Translator()

    # Save original functions
    original_xlabel = plt.xlabel
    original_ylabel = plt.ylabel
    original_title = plt.title

    # Define a wrapper for translation
    def translate_text(func):
        def wrapper(text, *args, **kwargs):
            translated_text = translator.translate(
                text, src='auto', dest=dest).text
            return func(translated_text, *args, **kwargs)
        return wrapper

    # Redefine functions globally
    plt.xlabel = translate_text(original_xlabel)
    plt.ylabel = translate_text(original_ylabel)
    plt.title = translate_text(original_title)

    # Function to translate tick labels
    def translate_tick_labels(ax):
        ax.set_xticklabels(
            [translator.translate(label.get_text(), src='auto', dest=dest).text
             for label in ax.get_xticklabels()]
        )
        ax.set_yticklabels(
            [translator.translate(
                label.get_text(), src='auto', dest=dest).text
             for label in ax.get_yticklabels()]
        )

    # Function to translate legend labels
    def translate_legend_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        translated_labels = [
            translator.translate(
                label, src='auto', dest=dest).text for label in labels
        ]
        ax.legend(handles, translated_labels)

    # Extend `plt.gca` to include tick and legend translation
    original_gca = plt.gca

    def custom_gca(*args, **kwargs):
        ax = original_gca(*args, **kwargs)
        translate_tick_labels(ax)
        translate_legend_labels(ax)
        return ax

    plt.gca = custom_gca


def generate_code_book(keys, values, filename=None):
    norm_keys = [normalize_string(k) for k in keys]
    data_dict = dict(zip(norm_keys, values))
    data_dict['_filename'] = filename

    return SimpleNamespace(**data_dict)


class ExtendedNamespace(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def items(self):
        return self.__dict__.items()


def list_to_codebook(list_of_strings):
    return SimpleNamespace(**dict([(normalize_string(n), n)
                                   for n in list_of_strings]))


def df_to_codebook(df):
    def get_answer_mapping(column):
        # Create a mapping for categorical columns (normalized -> actual value)
        if isinstance(df[column].dtype, pd.CategoricalDtype) or df[column].dtype == 'object':
            return SimpleNamespace(**dict([(normalize_string(str(val)), val) for val in df[column].unique()]))
        return None

    # Build the codebook
    return SimpleNamespace(
        **{
            normalize_string(col): SimpleNamespace(
                v=col,  # Original column name
                answer=get_answer_mapping(col)  # Mapping for categorical columns
            )
            for col in df.columns
        }
    )


def set_global_bw_style():
    """
    Configures Matplotlib & Seaborn globally for true black-and-white plots:
      - White background, no grid
      - One or more shades of black/gray for all lines
      - Distinct linestyles for differentiation
    """
    # (A) Choose purely black OR a few shades of gray if you want variety
    # Using multiple "black" entries with different linestyles ensures each line is black
    # but visually distinct by dashes.
    tab10_cmap = plt.cm.get_cmap('tab10')
    tab10_colors = [tab10_cmap(i) for i in range(6)]
    # bw_colors = ["black"] * 6   # 6 lines, all black
    # Or, if you prefer some gray shades for large numbers of categories:
    # bw_colors = ["black", "dimgray", "gray", "darkgray", "lightgray", "silver"]

    # (B) Define 6 clearly distinct linestyles
    # Adjust as needed (remove or add if you have more or fewer lines)
    linestyles = [
        '-',             # Solid
        '--',            # Dashed
        '-.',            # Dash-dot
        ':',             # Dotted
        (0, (10, 5)),    # Long dash
        (0, (5, 2, 1, 2))# Dash-dot-dot
    ]

    # Create a combined cycler for color and linestyle
    # This gives each successive line a new linestyle, but keeps color black (or gray).
    style_cycler = (
        cycler(color=tab10_colors)[:len(linestyles)] +
        cycler(linestyle=linestyles)
    )

    # Update Matplotlib's rcParams to use our custom cycle
    plt.rcParams.update({
        "axes.prop_cycle": style_cycler,
        "axes.edgecolor": "black",  # Crisp axes
        "axes.linewidth": 0.8,
        "grid.alpha": 0,            # Disable grid
        "lines.linewidth": 1.5      # Thicker lines for clarity
    })

    # Tell Seaborn to:
    #  - have a white background ("white")
    #  - ignore its default palette by passing palette=None
    #  - inherit our custom axes.prop_cycle
    sns.set_theme(
        style="white",
        palette=None,  # Important: prevents Seaborn from overriding with its own color palette
        rc={
            "axes.prop_cycle": plt.rcParams["axes.prop_cycle"],
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.alpha": 0  # No grid
        }
    )


set_global_bw_style()
