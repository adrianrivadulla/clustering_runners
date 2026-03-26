def compare_0D_contvar_indgroups_one_condition(datadict, grouping, **kwargs):
    """
    Compare continuous variables between independent groups using various statistical tests.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    grouping (list or np.ndarray): List or array containing the group labels for each data point.
    title_kword (str): Keyword to be used in the title of the plots.
    figdir (str): Directory where the plots will be saved.
    colours (list or np.ndarray): List or array containing the colors for the groups.

    Returns:
    disc_comp (dict): A dictionary containing the results of the statistical tests.
    figs (dict): A dictionary containing the figures generated for normality checks.
    """

    # Get kwargs
    colours = kwargs.get("colours", sns.color_palette("Set2", len(np.unique(grouping))))
    group_names = kwargs.get("group_names", np.unique(grouping))

    disc_comp = {}
    figs = {}

    for key, values in datadict.items():
        disc_comp[key] = {}

        # Check for nans
        if np.any(np.isnan(values)):
            print(f"NaNs found in {key} and they will be removed.")

        # Get variable in groups
        holder = pd.DataFrame({key: np.squeeze(values)})
        holder["grouping"] = grouping
        groups = [
            holder.groupby(["grouping"]).get_group(x)[key].dropna()
            for x in np.sort(holder["grouping"].dropna().unique())
        ]

        # Run normality tests
        disc_comp[key]["normality"] = {}

        figs[key], axes = plt.subplots(1, len(groups))
        figs[key].set_size_inches([11, 3.3])

        # test trigger
        param_route = 1

        for labi, group in enumerate(groups):
            disc_comp[key]["normality"][str(labi)] = {}
            (
                disc_comp[key]["normality"][str(labi)]["W_stat"],
                disc_comp[key]["normality"][str(labi)]["p"],
            ) = stats.shapiro(group)

            # if there were violations of normality or homoscedasticity change trigger for tests later
            if disc_comp[key]["normality"][str(labi)]["p"] <= 0.05:
                param_route = 0

            # Q-Q plots
            sm.qqplot(
                group,
                ax=axes[labi],
                markeredgecolor=colours[labi],
                markerfacecolor=colours[labi],
                line="r",
            )

            # This is so goofy but sm.qqplot doesn't take a line colour argument and I need to change it here
            axes[labi].get_lines()[1].set_color("black")

            # Set labels and title
            axes[labi].set_xlabel(group_names[labi])

            if disc_comp[key]["normality"][str(labi)]["p"] < 0.001:
                axes[labi].set_title(
                    "W: "
                    + str(np.round(disc_comp[key]["normality"][str(labi)]["W_stat"], 3))
                    + "; p < 0.001"
                )
            else:
                axes[labi].set_title(
                    "W: "
                    + str(np.round(disc_comp[key]["normality"][str(labi)]["W_stat"], 3))
                    + "; p = "
                    + str(np.round(disc_comp[key]["normality"][str(labi)]["p"], 3))
                )

        figs[key].suptitle(key)
        plt.tight_layout()

        # Parametric route
        if param_route:
            if len(groups) == 2:
                # Run heteroscedasticity tests
                disc_comp[key]["homoscedasticity"] = {}
                (
                    disc_comp[key]["homoscedasticity"]["Levene_stat"],
                    disc_comp[key]["homoscedasticity"]["p"],
                ) = stats.levene(*groups)

                if disc_comp[key]["homoscedasticity"]["p"] > 0.05:
                    # Independent standard t-test
                    disc_comp[key]["ttest_ind"] = {}
                    (
                        disc_comp[key]["ttest_ind"]["t"],
                        disc_comp[key]["ttest_ind"]["p"],
                    ) = stats.ttest_ind(*groups)

                else:
                    # Welch's t-test
                    disc_comp[key]["ttest_ind"] = {}
                    (
                        disc_comp[key]["ttest_ind"]["welch_t"],
                        disc_comp[key]["ttest_ind"]["p"],
                    ) = stats.ttest_ind(*groups, equal_var=False)

                # Get Cohen's d
                disc_comp[key]["ttest_ind"]["Cohens_d"] = (
                    np.mean(groups[0]) - np.mean(groups[1])
                ) / np.sqrt(
                    (np.std(groups[0], ddof=1) ** 2 + np.std(groups[1], ddof=1) ** 2)
                    / 2
                )

                # Get Hedge's g
                disc_comp[key]["ttest_ind"]["Hedges_g"] = disc_comp[key]["ttest_ind"][
                    "Cohens_d"
                ] * (1 - (3 / (4 * (len(groups[0]) + len(groups[1]) - 2) - 1)))

            elif len(groups) > 2:
                # One-way ANOVA
                disc_comp[key]["ANOVA_1"] = {}
                disc_comp[key]["ANOVA_1"]["F_stat"], disc_comp[key]["ANOVA_1"]["p"] = (
                    stats.f_oneway(*groups)
                )

                if disc_comp[key]["ANOVA_1"]["p"] <= 0.05:
                    # Bonferroni post hoc tests
                    disc_comp[key]["Bonferroni_post_hoc"] = posthoc_ttest(
                        groups, p_adjust="bonferroni"
                    )

        # Non-parametric route
        else:
            if len(groups) == 2:
                # Mann-Whitney U test
                disc_comp[key]["mann_whitney_U"] = {}
                (
                    disc_comp[key]["mann_whitney_U"]["U_stat"],
                    disc_comp[key]["mann_whitney_U"]["p"],
                ) = stats.mannwhitneyu(*groups)

            elif len(groups) > 2:
                # Kruskal
                disc_comp[key]["Kruskal"] = {}
                disc_comp[key]["Kruskal"]["Hstat"], disc_comp[key]["Kruskal"]["p"] = (
                    stats.kruskal(*groups)
                )

                if disc_comp[key]["Kruskal"]["p"] <= 0.05:
                    # Dunn post hoc tests
                    disc_comp[key]["Dunn_post_hoc"] = posthoc_dunn(
                        groups, p_adjust="bonferroni"
                    )

    return disc_comp, figs
