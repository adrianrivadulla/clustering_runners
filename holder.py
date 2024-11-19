import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def multispeed_kinematics_comparison(clustdata, stages, speeds, discvars, contvars, savingkw, reportdir):


    # Initialise stat_comparison
    stat_comparison = {'0D': {}, '1D': {}}

    # 0D variables: 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    for vari, varname in enumerate(discvars):
        discvarfig, discvaraxs = plt.subplots(1, 3, figsize=(11, 3))
        discvaraxs = discvaraxs.flatten()

        stat_comparison['multispeed']['0D'][varname] = {}

        # Get data
        df = pd.DataFrame()
        df[varname] = np.concatenate(clustdata['multispeed'][varname].T)
        df['speed'] = np.concatenate(
            [[int(speeds[stgi])] * clustdata['multispeed'][varname].shape[0] for stgi, stage in enumerate(stages)])
        df['clustlabel'] = np.tile(clustdata['multispeed']['ptlabels']['clustlabel'].values, len(stages))
        df['ptcode'] = np.tile(clustdata['multispeed']['ptlabels']['ptcode'].values, len(stages))

        # Run 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
        stat_comparison['multispeed']['0D'][varname] = anova2onerm_0d_and_posthocs(df,
                                                                                   dv=varname,
                                                                                   within='speed',
                                                                                   between='clustlabel',
                                                                                   subject='ptcode')

        # Plot results
        for speedi, speed in enumerate(speeds):

            # Violin plot
            sns.violinplot(ax=discvaraxs[speedi],
                           x='clustlabel',
                           y=varname,
                           data=df.loc[df['speed'] == speed],
                           palette=uniqclustcolours,
                           legend=False)

            # Xticks
            discvaraxs[speedi].set_xticklabels([f'C{int(x)}' for x in discvaraxs[speedi].get_xticks()])

            # Add stats in xlabel
            if stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['p-unc'].loc[
                stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm']['Source'] == 'clustlabel'].values < 0.05:
                statsstr = write_0Dposthoc_statstr(stat_comparison['multispeed']['0D'][varname]['posthocs'],
                                                   'speed * clustlabel', 'speed', speed)
                discvaraxs[speedi].set_xlabel(f'C: {statsstr}', fontsize=11)

            else:
                discvaraxs[speedi].set_xlabel(' ', fontsize=11)

            # y label
            if speedi == 0:
                discvaraxs[speedi].set_ylabel(kinematics_ylabels[varname])
            else:
                discvaraxs[speedi].set_ylabel('')

            # Add title
            discvaraxs[speedi].set_title(f'{speed} km/h')

        # Same y limits
        ylims = [ax.get_ylim() for ax in discvaraxs]
        for ax in discvaraxs:
            ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

        # Set suptitle as the var name and stats
        statsstr = write_0DmixedANOVA_statstr(stat_comparison['multispeed']['0D'][varname]['ANOVA2onerm'],
                                              between='clustlabel',
                                              within='speed',
                                              betweenlabel='C',
                                              withinlabel='S')

        discvarfig.suptitle(f'{kinematics_titles[varname]}\n{statsstr}')

        # Save and close
        plt.tight_layout()
        discvarfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_{varname}_ANOVA2onerm.png'), dpi=300,
                           bbox_inches='tight')
        plt.close(discvarfig)

    # 1D variables: SPM 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    speedfig, speedaxs = plt.subplots(2, 3, figsize=(11, 4.5))
    speedaxs = speedaxs.flatten()

    # Get avge toe off for each speed and for each cluster based on duty factor for the plots
    avgeto = {}
    speedavgeto = []
    for stage in stages:
        avgeto[stage] = []
        for clusti, uniqclust in enumerate(uniqclustlabels):
            clustidcs = np.where(clustdata[stage]['ptlabels']['clustlabel'] == uniqclust)[0]
            avgeto[stage].append(np.round(np.mean(clustdata[stage]['DUTYFACTOR'][clustidcs, :]) * 100, 1))

        speedavgeto.append(np.round(np.mean(clustdata[stage]['DUTYFACTOR']) * 100, 1))

    for vari, contvar in enumerate(contvars):

        stat_comparison['multispeed']['1D'][contvar] = {}

        # Initialise data holder
        group = []
        speed = []
        subject = []
        Y = []
        Ydiff = []

        # Prepare data for SPM and SPM mean and std plots
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        basegrid = fig.add_gridspec(2, 1)
        topgrid = basegrid[0].subgridspec(1, len(stages))
        bottomgrid = basegrid[1].subgridspec(1, len(stages) - 1)

        upperaxs = []
        loweraxs = []

        for stgi, stage in enumerate(stages):

            # Append group speed and subject
            group.append(clustdata['multispeed']['ptlabels']['clustlabel'].values)
            speed.append(np.ones(clustdata['multispeed'][varname].shape[0]) * stgi)
            subject.append(np.arange(len(clustdata['multispeed']['ptlabels']['clustlabel'].values)))
            Y.append(clustdata[stage][contvar])

            # Create axis
            upperaxs.append(fig.add_subplot(topgrid[0, stgi]))

            # Plot mean and std curves
            for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                # Top row: group by group for each speed
                spm1d.plot.plot_mean_sd(Y[-1][np.where(group[-1] == lab)[0], :],
                                        x=np.linspace(0, 100, Y[-1].shape[1]),
                                        linecolor=uniqclustcolours[labi], facecolor=uniqclustcolours[labi],
                                        ax=upperaxs[stgi])

            # Add vertical line at avge toe off (outside the previous loop so we can get the final ylimits)
            for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                upperaxs[stgi].axvline(x=avgeto[stage][labi], color=uniqclustcolours[labi], linestyle=':')

            # xlabel. This ensures they are all the same size and will get filled with stats if post-hocs were performed
            upperaxs[stgi].set_xlabel(' ')

            # Speed figure
            spm1d.plot.plot_mean_sd(Y[-1], x=np.linspace(0, 100, Y[-1].shape[1]),
                                    linecolor=speedcolours[stgi], facecolor=speedcolours[stgi],
                                    ax=speedaxs[vari])

            if stgi > 0:

                # Calculate change from one speed to another
                Ydiff.append(Y[-1] - Y[-2])

                # Plot it by group
                loweraxs.append(fig.add_subplot(bottomgrid[0, stgi - 1]))

                # Add horizontal line at 0
                loweraxs[-1].axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=1)

                for uni in np.sort(np.unique(group)):
                    spm1d.plot.plot_mean_sd(Ydiff[-1].T[:, group[stgi] == uni].T,
                                            x=np.linspace(0, 100, Ydiff[-1].shape[1]),
                                            linecolor=colours[uni], facecolor=colours[uni],
                                            ax=loweraxs[-1])

                # Add vline at avge toe off between speeds (outside the previous loop so we can get the final ylimits)
                for labi, lab in enumerate(np.sort(np.unique(group[-1]))):
                    loweraxs[-1].axvline(x=np.mean([avgeto[stages[stgi - 1]][labi], avgeto[stage][labi]]),
                                         color=uniqclustcolours[labi], linestyle=':')

                # Set title
                loweraxs[-1].set_title(f'{speeds[stgi]} wrt {speeds[stgi - 1]} km/h')

                # xlabel. This ensures they are all the same size and
                #  will get filled with stats if post-hocs were performed
                loweraxs[-1].set_xlabel(' ')

                # Legend
                loweraxs[-1].legend(['_nolegend_', 'C0', 'C1'],
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 0),
                                    ncol=2,
                                    bbox_transform=fig.transFigure,
                                    frameon=False)
                plt.subplots_adjust(bottom=0.11)

            # Title
            upperaxs[stgi].set_title(stgtitles[stgi])

        # ylabels
        upperaxs[0].set_ylabel(kinematics_ylabels[contvar])
        loweraxs[0].set_ylabel('${\Delta}$')

        # Get ylims for all upperaxs
        ylims = [ax.get_ylim() for ax in upperaxs]

        # Set ylims for all upperaxs
        for ax in upperaxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # Get ylims for all loweraxs
        ylims = [ax.get_ylim() for ax in loweraxs]

        # Set ylims for all loweraxs
        for ax in loweraxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # add vertical lines at avge toe off to speed figure
        for spavgetoi, spavgeto in enumerate(speedavgeto):
            speedaxs[vari].axvline(x=spavgeto, color=speedcolours[spavgetoi], linestyle=':')

        # title and ylabel for speed figure
        speedaxs[vari].set_title(kinematics_titles[contvar])
        speedaxs[vari].set_ylabel(kinematics_ylabels[contvar])

        # Conduct SPM analysis
        spmlist = spm1d.stats.nonparam.anova2onerm(np.concatenate(Y, axis=0),
                                                   np.concatenate(group),
                                                   np.concatenate(speed),
                                                   np.concatenate(subject))
        stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'] = spmlist.inference(alpha=0.05, iterations=1000)

        # Post hoc tests and figures
        stat_comparison['multispeed']['1D'][contvar]['posthocs'] = {}

        # Add patches to speed figure if there is an effect of speed
        if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][1].h0reject:
            # Scaler for sigcluster endpoints
            tscaler = speedaxs[vari].get_xlim()[1] / (Y[0].shape[1] - 1)

            # Add patches to speed figure
            add_sig_spm_cluster_patch(speedaxs[vari], stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][1],
                                      tscaler=tscaler)

        # Add title to speed figure
        statstr = f'F* = {write_spm_stats_str(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][1], mode="stat")}'

        speedaxs[vari].set_title(f'{kinematics_titles[contvar]}\n{statstr}')

        # Follow up with post-hoc tests if cluster effects are found
        if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][0].h0reject:

            stat_comparison['multispeed']['1D'][contvar]['posthocs']['cluster'] = {}

            # For each speed
            for spi, (groupi, Yi) in enumerate(zip(group, Y)):

                stat_comparison['multispeed']['1D'][contvar]['posthocs']['cluster'][stages[spi]] = {}

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Yi[groupi == 0, :], Yi[groupi == 1, :], )
                snpmi = snpm.inference(alpha=0.05 / len(Y), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison['multispeed']['1D'][contvar]['posthocs']['cluster'][stages[spi]]['snpm_ttest2'] = snpmi

                # Add stats to xlabel
                statstr = f't* = {write_spm_stats_str(snpmi, mode="full")}'
                upperaxs[spi].set_xlabel(statstr, fontsize=10)

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{contvar}_posthoc_{stages[spi]}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir, f'{savingkw}_{contvar}_posthoc_{stages[spi]}.png'))
                plt.close(plt.gcf())

                # Add patches to upperaxs if significant diffs are found
                if snpmi.h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = upperaxs[spi].get_xlim()[1] / (Y[0].shape[1] - 1)

                    # Add significant pathces to upperaxs
                    add_sig_spm_cluster_patch(upperaxs[spi], snpmi, tscaler=tscaler)

        # Interaction effect
        if stat_comparison['multispeed']['1D'][contvar]['ANOVA2onerm'][2].h0reject:

            stat_comparison['multispeed']['1D'][contvar]['posthocs']['interaction'] = {}

            # Calculate change in conditions
            for condi in range(len(stages) - 1):

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Ydiff[condi][group[condi] == 0, :],
                                                   Ydiff[condi][group[condi] == 1, :], )
                snpmi = snpm.inference(alpha=0.05 / len(range(len(stages) - 1)), two_tailed=True, iterations=1000)

                # Add snpmi to dictionary
                stat_comparison['multispeed']['1D'][contvar]['posthocs']['interaction'][
                    f'{speeds[condi + 1]}_wrt_{speeds[condi]}'] = {}
                stat_comparison['multispeed']['1D'][contvar]['posthocs']['interaction'][
                    f'{speeds[condi + 1]}_wrt_{speeds[condi]}']['snpm_ttest2'] = snpmi

                # Add stats to xlabel
                statstr = f't* = {write_spm_stats_str(snpmi, mode="full")}'
                loweraxs[condi].set_xlabel(statstr, fontsize=10)

                # Plot
                plt.figure()
                snpmi.plot()
                snpmi.plot_threshold_label(fontsize=8)
                snpmi.plot_p_values(size=10)
                plt.gcf().suptitle(f'{contvar}_posthoc_{speeds[condi + 1]}_v_{speeds[condi]}')

                # Save figure and close it
                plt.savefig(os.path.join(reportdir,
                                         f'{savingkw}_multispeed_{contvar}_interact_posthoc_{speeds[condi + 1]}_v_{speeds[condi]}.png'))
                plt.close(plt.gcf())

                # Add patches to loweraxs if significant diffs are found
                if snpmi.h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = loweraxs[condi].get_xlim()[1] / (Ydiff[0].shape[1] - 1)

                    # Add significant pathces to loweraxs
                    add_sig_spm_cluster_patch(loweraxs[condi], snpmi, tscaler=tscaler)

        # Write cluster effect string for suptitle
        statstr = f'C: F* = {write_spm_stats_str(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][0], mode="full")}'

        # Write interaction effect string for suptitle
        statstr += f'; CxS: F* = {write_spm_stats_str(stat_comparison["multispeed"]["1D"][contvar]["ANOVA2onerm"][2], mode="full")}'

        # Set suptitle
        fig.suptitle(f'{kinematics_titles[contvar]}\n{statstr}')

        # Save and close
        plt.subplots_adjust(bottom=0.13)
        plt.tight_layout()
        fig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_{contvar}_ANOVA2onerm.png'), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # Add legend to last plot in speed figure
    speedaxs[-1].legend(speeds, frameon=False)

    # Save speed figure
    plt.tight_layout()
    speedfig.savefig(os.path.join(reportdir, f'{savingkw}_multispeed_kinematics_by_speed.png'), dpi=300,
                     bbox_inches='tight')
    plt.close(speedfig)
