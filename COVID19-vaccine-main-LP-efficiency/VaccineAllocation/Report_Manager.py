import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

base_path = Path(__file__).parent

percentiles = [5, 50, 95, 99, 100]
tier_colors = ["green", "blue", "yellow", "orange", "red"]


class Report:
    """
    Build a latex report for city statistics.
    """

    def __init__(self,
                 instance,
                 sim_data,
                 policy_data,
                 stats_start_date=None,
                 stats_end_date=None,
                 central_path=0,
                 template_file="report_template.tex"):

        self.instance = instance
        self.sim_data = sim_data
        self.policy_data = policy_data
        self.central_path = central_path
        self.path_to_report = base_path / "reports"
        self.template_file = f"{self.path_to_report}/{template_file}"
        self.stats_start_date = stats_start_date
        self.stats_end_date = stats_end_date
        self.T_start = instance.cal.calendar.index(stats_start_date)
        self.T_end = instance.cal.calendar.index(stats_end_date)
        self.report_data = {}
        self.cap_list = {'IHT': [self.instance.hosp_beds], "ICU": [350, 300, 250, 200, 150]}

    def build_report(self):
        """
        Calculate the statistics in self.report_data dictionary. Update the latex template with the calculated
        values. The keys should be consistent with the latex template.
        :return:
        """
        # Clean up the data:
        for key, data in self.sim_data.items():
            self.sim_data[key] = np.array([np.sum(s, axis=(1, 2))[self.T_start:self.T_end] for s in data])
        self.sim_data["IHT_history"] = self.sim_data["ICU_history"] + self.sim_data["IH_history"]

        # Initialize the city report:
        self.report_data = {'instance_name': self.instance.city,
                            'CITY': self.instance.city,
                            'STATISTICS-START-DATE': self.stats_start_date.strftime("%Y-%m-%d"),
                            'STATISTICS-END-DATE': self.stats_end_date.strftime("%Y-%m-%d")}

        # Add hospitalizations stats to the report
        self.hospital_stats()
        # Add tier statistics to the report:
        self.tier_stats()
        # Update the template latex file with statistics and create the new city file and run:
        self.generate_report()

    def hospital_stats(self):
        """
        Calculate city statistics on hospital peaks, ICU peaks, capacity, total deaths etc.
        """
        print('Hospitalization Peaks')
        for key in ["ICU", 'IHT']:
            peak_days = np.argmax(self.sim_data[f"{key}_history"], axis=1)
            peak_vals = np.take_along_axis(self.sim_data[f"{key}_history"], peak_days[:, None], axis=1)
            peak_mean = np.mean(peak_vals)
            self.report_data[f"MEAN-{key.upper()}-PEAK"] = np.round(peak_mean)

            for q in percentiles:
                peak_day_percentile = int(np.round(np.percentile(peak_days, q)))
                peak_percentile = np.percentile(peak_vals, q)
                self.report_data[f"{key.upper()}-PEAK-P{q}"] = np.round(peak_percentile)
                self.report_data[f"{key.upper()}-PEAK-DATE-P{q}"] = self.instance.cal.calendar[
                    peak_day_percentile + self.T_start].strftime(
                    "%Y-%m-%d")

            # Patients after capacity
            for cap in self.cap_list[key]:
                patients_excess = np.sum(np.maximum(self.sim_data[f"{key}_history"][:, :-1] - cap, 0), axis=1)
                self.report_data[f'{cap}-PATHS-{key.upper()}-UNMET'] = 100 * np.round(
                    np.sum(patients_excess > 0) / len(self.sim_data[f"{key}_history"]), 3)
                self.report_data[f'{cap}-MEAN-{key.upper()}-UNSERVED'] = np.round(patients_excess.mean())
                self.report_data[f'SD-{key.upper()}-UNSERVED'] = np.round(patients_excess.std())
                for q in percentiles:
                    self.report_data[f'{cap}-{key.upper()}-UNSERVED-P{q}'] = np.round(np.percentile(patients_excess, q))

        # Deaths data
        self.report_data['MEAN-DEATHS'] = np.round(
            np.mean(self.sim_data['D_history'][:, -1] - self.sim_data['D_history'][:, 0]), 0)
        for q in percentiles:
            self.report_data[f'P{q}-DEATHS'] = np.round(
                np.percentile(self.sim_data['D_history'][:, -1] - self.sim_data['D_history'][:, 0], q))

    def tier_stats(self):
        """
        Calculate:
            - number of days spent in each tier,
            - number of paths having that particular tier.
        and update the city report.
        """
        tier_hist_list = []
        lockdown_report = []
        for u in range(1, len(tier_colors)):
            tier_hist = np.array([np.sum(np.array(hist) == u) for hist in self.policy_data["tier_history"]])

            lockdown_report.append({
                f'MEAN-{tier_colors[u].upper()}': f'{np.mean(tier_hist):.2f}',
                f'P50-{tier_colors[u].upper()}': f'{np.percentile(tier_hist, q=50)}',
                f'P5-{tier_colors[u].upper()}': f'{np.percentile(tier_hist, q=5)}',
                f'P95-{tier_colors[u].upper()}': f'{np.percentile(tier_hist, q=95)}'
            })
            self.report_data.update(lockdown_report[u - 1])
            self.report_data[f'PATHS-IN-{tier_colors[u].upper()}'] = 100 * round(
                sum(tier_hist > 0) / len(tier_hist), 4)
            tier_hist_list.append(tier_hist)

        self.report_data['BAR_PLOT'] = self.plot_tier_bar(tier_hist_list)

    def generate_report(self):
        """
        Generate a new city report form the latex template and run the latex file.
        :return:
        """
        instance_name = self.report_data['instance_name']
        tex_file = str(
            self.path_to_report / f'report_{instance_name}_{self.trigger_summary()}_{self.stats_start_date.date()}.tex')

        pdf_file = f'report_{instance_name}_{self.trigger_summary()}_{self.stats_start_date.date()}.pdf'
        if not self.path_to_report.exists():
            os.system(f"mkdir {self.path_to_report}")

        self.fill_template(self.template_file, tex_file)
        os.system(f"pdflatex {tex_file}")  # Run the latex file to generate pdf.
        os.system(f"mv {pdf_file} {self.path_to_report}")  # move the pdf file to the correct directory for reports.

        # Delete the extra files:
        os.remove(f'report_{instance_name}_{self.trigger_summary()}_{self.stats_start_date.date()}.aux')
        os.remove(f'report_{instance_name}_{self.trigger_summary()}_{self.stats_start_date.date()}.log')

    def fill_template(self, template_file, report_file):
        """
        Fill the template with the calculated key statistics.
        :param template_file: the template file name.
        :param report_file: the new file name for our city statistics.
        :return:
        """
        with open(template_file, 'r') as file:
            tex_template = file.read()

        for k, v in self.report_data.items():
            tex_template = tex_template.replace(k, str(v), 1)

        tex_template = tex_template.replace('INSERT-TRIGGER-DESCRIPTION', self.trigger_summary())

        with open(report_file, 'w') as out_rep:
            out_rep.write(tex_template)

    def trigger_summary(self):
        """
        Create the name of the report according to the stage alert system that is used.
        """
        if "case_threshold" in self.policy_data.keys():
            return f"CDC-{self.policy_data['case_threshold']}"
        else:
            return f"Staged Alert-{self.policy_data['lockdown_thresholds']}"

    def plot_tier_bar(self, tier_hist_list: list):
        """
        Plots histograms of average proportion of days each tier was active.
        :param tier_hist_list: total number of days each tier was active for each sample paths.
        :return: the plot filename.
        """
        tier_hist_mean = [np.mean(tier) / (self.T_end - self.T_start) for tier in tier_hist_list]
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3, 4], tier_hist_mean, color=tier_colors[1:])
        ax.set_ylabel("Proportion of days")
        ax.set_ylim(0, 1)
        plot_filename = self.path_to_report / f"bar_{self.report_data['instance_name']}_{self.trigger_summary()}_{self.stats_start_date.date()}.pdf"
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(plot_filename)
        return plot_filename
