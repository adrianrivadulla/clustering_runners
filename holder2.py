append_bottom_leaves_dendrogram(dendroax, labelcolour=ptlabels[stages[stgi - 1]])

# Transition analysis from previous to current partition
trans_analysis = TransitionAnalysis(ptlabels[stages[stgi - 1]], colourid, dendroax)
trans_analysis.dendroax.set_title(f'{stgtitles[stgi]} '
                                  f'(Silh = {np.round(score_table.loc["Silhouette"].values[0], 3)}, '
                                  f'AMI = {np.round(trans_analysis.ami, 3)})')

# Write legend. This is currently disgusting but it seems to work

# Temporary legend
templegend = trans_analysis.currentcolourlabels['colourcode'].to_list()
templegend.insert(0, '_nolegend')
legend = trans_analysis.dendroax.legend(templegend, frameon=False)

# Get handles and their colours
handles = legend.legendHandles
handlecolours = [handle.get_color() for handle in handles]

# Get labels which should be the same as the CN colourcode in matplotlib
leglabels = [label.get_text() for label in legend.get_texts()]
leglabelcolours = [mcolors.to_rgba(leglabel) for leglabel in leglabels]

# Get indices of leglabelcolours in handlecolours
idcs = []
for handlecolour in handlecolours:
    for leglabi, leglabelcolour in enumerate(leglabelcolours):
        if np.all(handlecolour == leglabelcolour):
            idcs.append(leglabi)
            break

# Get the colourcode again and order them correctly
finalcolours = trans_analysis.currentcolourlabels['colourcode'].to_list()
orderedcolours = [finalcolours[idx] for idx in idcs]

# Get count of pts in each cluster
clustcount = [len(trans_analysis.curr_colourid.loc[trans_analysis.curr_colourid['colourcode'] == colourlabel])
              for
              colourlabel in orderedcolours]

# Subtract 1 from the digit in finalcolours
orderedcolours = [orderedcolours[:-1] + str(int(orderedcolours[-1]) - 1) for orderedcolours in orderedcolours]

# Add count of pts in each cluster
orderedcolours = [f'{orderedcolour} ({clustcounti})' for orderedcolour, clustcounti in
                  zip(orderedcolours, clustcount)]

# Set the legend correctly now
orderedcolours.insert(0, '_nolegend')
trans_analysis.dendroax.legend(orderedcolours, frameon=False)

