

def add_suffix_to_titles(fig, suffix):
    """
    """

    for ax in fig.axes:
        currtitle = ax.get_title()
        lines = currtitle.split('\n')
        newtitle = f'{lines[0]}{suffix}'
        if len(lines) > 1:
            newtitle += f'\n{lines[1]}'
        ax.set_title(newtitle)
