from matplotlib import pyplot as plt, colors
from pathlib import Path
import numpy as np
from datetime import datetime as dt
import calendar as py_cal

base_path = Path(__file__).parent
######################################################################################
compartment_names = {
    'ToIHT_history': 'COVID-19 Hospital Admissions\n(Seven-day Average)',
    'ToIHT_history_sum': 'COVID-19 Hospital Admissions per 100k\n(Seven-day Sum)',
    'IH_history_average': 'Percent of Staffed Inpatient Beds\n(Seven-day Average)',
    'D_history': 'Deaths',
    'R_history': 'Recovered',
    'ICU_history': 'COVID-19 ICU Patients',
    'IH_history': 'COVID-19 Hospitalizations',
    'ToSS_unvax': 'Evasion of Vaccine Induced Immunity',
    'ToRS_unvax': 'Evasion of Natural Immunity',
    'R_history': 'Recovered',
    'ToIY_history': 'COVID-19 New Symptomatic Cases per 100k\n(Seven-day Sum)'
}
y_lim = {'ToIHT_history': 150,
         'ToIHT_history_sum': 60,
         'IH_history_average': 1,
         'D_history': 3000,
         'ICU_history': 300,
         'IH_history': 500,
         'ToSS_unvax': 4000,
         'ToRS_unvax': 3000,
         'R_history': 500000,
         'ToIY_history': 1500}

plt.rcParams["font.size"] = "18"


######################################################################################
# Plotting Module


class Plot:
    """
    Plots a list of sample paths in the same figure with different plot backgrounds.
    """

    def __init__(self, instance, policy_data, real_history_end_date, real_data, sim_data, var, color='teal',
                 text_size=28):
        self.instance = instance
        self.real_history_end_date = real_history_end_date
        self.policy_data = policy_data
        self.real_data = real_data
        self.sim_data = np.sum(sim_data, axis=(1, 2))
        self.var = var
        self.T = len(np.sum(sim_data, axis=(1, 2)))
        self.T_real = (real_history_end_date - instance.start_date).days
        self.y_lim = y_lim
        self.text_size = text_size
        self.base_plot(color)

    def base_plot(self, color):
        """
        The base plotting function sets the common plot design for different type of plots.
        :return:
        """
        self.path_to_plot = base_path / "plots"

        self.fig, (self.ax1, self.actions_ax) = plt.subplots(2, 1, figsize=(17, 9),
                                                             gridspec_kw={'height_ratios': [10, 1.1]})
        self.policy_ax = self.ax1.twinx()

        if 'Seven-day Average' in compartment_names[self.var]:
            self.moving_avg()
        elif 'Seven-day Sum' in compartment_names[self.var]:
            self.moving_sum()

        if self.real_data is not None:
            real_h_plot = self.ax1.scatter(range(self.T_real), self.real_data[0:self.T_real], color='maroon',
                                           zorder=100, s=15)

        self.ax1.plot(range(self.T), self.sim_data, color)

        # plot a vertical line to separate history from projections:
        self.ax1.vlines(self.T_real, 0, y_lim[self.var], colors='k', linewidth=3)

        # Plot styling:
        # Axis limits:
        self.ax1.set_ylim(0, y_lim[self.var])
        self.policy_ax.set_ylim(0, 1)
        self.ax1.set_ylabel(compartment_names[self.var])
        self.actions_ax.set_xlim(0, self.T)
        self.set_x_axis()
        # Order of layers
        self.ax1.set_zorder(self.policy_ax.get_zorder() + 10)  # put ax in front of policy_ax
        self.ax1.patch.set_visible(False)  # hide the 'canvas'
        # Plot margins
        self.actions_ax.margins(0)
        self.ax1.margins(0)
        self.policy_ax.margins(0)

        # Axis ticks.
        if "Percent" in compartment_names[self.var]:
            self.ax1.yaxis.set_ticks(np.arange(0, 1.001, 0.2))
            self.ax1.yaxis.set_ticklabels(
                [f' {np.round(t * 100)}%' for t in np.arange(0, 1.001, 0.2)],
                rotation=0,
                fontsize=22)

        self.save_plot()

    def moving_avg(self):
        """
        Take the 7-day moving average of the data we are plotting.
        (Add percentage for percent of staffed inpatient beds).
        :return:
        """
        if self.var == 'IH_history_average':
            percent = self.instance.hosp_beds
        else:
            percent = 1
        n_day = self.instance.config["moving_avg_len"]

        self.sim_data = [self.sim_data[i: min(i + n_day, self.T)].mean() / percent for i in
                         range(self.T)]

        if self.real_data is not None:
            real_data = np.array(self.real_data)
            self.real_data = [real_data[0:self.T_real][i: min(i + n_day, self.T_real)].mean() / percent for
                              i in range(self.T_real)]

    def moving_sum(self):
        """
        Take the 7-day moving sum per 100k of the data we are plotting.
        :return:
        """
        n_day = self.instance.config["moving_avg_len"]
        total_population = np.sum(self.instance.N, axis=(0, 1))
        self.sim_data = [self.sim_data[i: min(i + n_day, self.T)].sum() * 100000 / total_population
                         for i in range(self.T)]

        if self.real_data is not None:
            real_data = np.array(self.real_data[0:self.T_real])
            self.real_data = [real_data[i: min(i + n_day, self.T_real)].sum() * 100000 / total_population
                              for i in range(self.T_real)]

    def set_x_axis(self):
        """
        Set the months and years on the x-axis of the plot.
        """
        # Axis ticks: write the name of the month on the x-axis:
        self.ax1.xaxis.set_ticks(
            [t for t, d in enumerate(self.instance.cal.calendar) if (d.day == 1 and d.month % 2 == 1)])
        self.ax1.xaxis.set_ticklabels(
            [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(self.instance.cal.calendar) if
             (d.day == 1 and d.month % 2 == 1)],
            rotation=0,
            fontsize=22)

        for tick in self.ax1.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
        self.ax1.tick_params(axis='y', labelsize=self.text_size, length=5, width=2)
        self.ax1.tick_params(axis='x', length=5, width=2)

        # Clean up the action_ax and policy_ax to write the years there:
        self.actions_ax.margins(0)
        self.actions_ax.spines['top'].set_visible(False)
        self.actions_ax.spines['bottom'].set_visible(False)
        self.actions_ax.spines['left'].set_visible(False)
        self.actions_ax.spines['right'].set_visible(False)
        self.policy_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False,
            labelright=False)  # labels along the bottom edge are off

        self.actions_ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

        year_list = {2020, 2021, 2022}  # fix this part later.
        year_ticks = {}
        for year in year_list:
            t1 = self.instance.cal.calendar.index(dt(year, 6, 15))
            t2 = self.instance.cal.calendar.index(dt(year, 2, 1)) if dt(year, 2,
                                                                        1) in self.instance.cal.calendar else self.T + 1
            if t1 <= self.T:
                year_ticks[year] = t1
            elif t1 > self.T >= t2:
                year_ticks[year] = t2
        # write down the year on the plot axis:
        for year in year_ticks:
            self.actions_ax.annotate(year,
                                     xy=(year_ticks[year], 0),
                                     xycoords='data',
                                     color='k',
                                     annotation_clip=True,
                                     fontsize=self.text_size - 2)

    def horizontal_plot(self, thresholds, tier_colors):
        """
        Plot the policy thresholds horizontally with corresponding policy colors.
        This plotting is  used with the indicator we keep track.
        For instance plot 7-day avg. hospital admission with the hospital admission thresholds in the background.

        Color the plot only for the part with projections where the transmission reduction is not fixed.
        """
        for id_tr, tr in enumerate(thresholds):
            u_color = tier_colors[id_tr]
            u_alpha = 0.6
            u_lb = tr
            u_ub = thresholds[id_tr + 1] if id_tr + 1 < len(thresholds) else y_lim[self.var]
            if u_lb >= -1 and u_ub >= 0:
                self.policy_ax.fill_between(range(self.T_real, self.T + 1),
                                            u_lb / y_lim[self.var],
                                            u_ub / y_lim[self.var],
                                            color=u_color,
                                            alpha=u_alpha,
                                            linewidth=0.0,
                                            step='pre')

        self.save_plot()

    def changing_horizontal_plot(self, surge_history, surge_states, thresholds, tier_colors):
        """
        Plot the policy thresholds horizontally with corresponding policy colors.
        This plotting is used when plotting the CDC staged-alert system. The thresholds change over time
        according to the case count.
        Color the plot only for the part with projections where the transmission reduction is not fixed.
        (I can combine this method with the horizontal plot later if necessary).
        """
        for u, state in enumerate(surge_states):
            fill = [True if s == u else False for s in surge_history[self.T_real:self.T]]
            for id_tr, tr in enumerate(thresholds[state]):
                u_color = tier_colors[id_tr]
                u_alpha = 0.6
                u_lb = tr
                u_ub = thresholds[state][id_tr + 1] if id_tr + 1 < len(thresholds[state]) else y_lim[self.var]
                if u_lb >= -1 and u_ub >= 0:
                    self.policy_ax.fill_between(range(self.T_real, self.T),
                                                u_lb / y_lim[self.var],
                                                u_ub / y_lim[self.var],
                                                color=u_color,
                                                alpha=u_alpha,
                                                linewidth=0.0,
                                                where=fill,
                                                step='pre')

        self.save_plot()

    def save_plot(self):
        plt.savefig(self.path_to_plot / f"{self.var}.png")

    def vertical_plot(self, tier_history, tier_colors):
        """
        Plot the historical policy vertically with corresponding policy colors.
        The historical policy can correspond to the five tiers or to the surge tiers in the CDC system.

        Color the plot only for the part with projections where the transmission reduction is not fixed.
        (We used to have a color decide tool to decide on the color of a transmission reduction level if
        it is in between to alert level. I can add that later if needed.)
        """
        for u in range(len(tier_colors)):
            u_color = tier_colors[u]
            u_alpha = 0.6
            fill = np.array(tier_history[self.T_real:self.T]) == u
            self.policy_ax.fill_between(range(self.T_real, self.T),
                                        0,
                                        1,
                                        where=fill,
                                        color=u_color,
                                        alpha=u_alpha,
                                        linewidth=1,
                                        step='pre')
        self.save_plot()

    def dali_plot(self):
        """
        Plot the tier history colors of different sample paths. I'll implement this plot when I moved to plotting
        multiple sample paths.
        :return:
        """
        pass
        # TODO: integrate the code for this later.
