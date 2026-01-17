"""
Statistical Analysis Module for SCIE Experiments

Provides comprehensive statistical analysis including:
- Descriptive statistics
- ANOVA (one-way, two-way, repeated measures)
- Post-hoc tests (Tukey HSD, Bonferroni)
- Effect sizes (Cohen's d, eta-squared)
- Non-parametric alternatives (Kruskal-Wallis, Mann-Whitney U)
- Confidence intervals
- Assumption tests (normality, homogeneity of variance)

All tests report p-values and effect sizes for SCIE publication standards.
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import math

# Statistical libraries
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import (
        ttest_ind, ttest_rel, f_oneway, kruskal,
        mannwhitneyu, wilcoxon, shapiro, levene,
        chi2_contingency, fisher_exact
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    pd = None
    warnings.warn("scipy not available - statistical analysis limited")

try:
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - some analyses limited")


@dataclass
class StatisticalResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float = 0.0
    effect_size_type: str = ""  # "d", "eta_squared", "r", etc.
    df: Any = None              # Degrees of freedom
    confidence_interval: Tuple[float, float] = None
    interpretation: str = ""
    significant: bool = False   # p < 0.05
    additional_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "df": str(self.df) if self.df else None,
            "confidence_interval": self.confidence_interval,
            "interpretation": self.interpretation,
            "significant": self.significant,
            "additional_info": self.additional_info
        }

    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f}{sig}, "
                f"{self.effect_size_type}={self.effect_size:.4f}")


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a group"""
    n: int
    mean: float
    std: float
    sem: float  # Standard error of mean
    median: float
    q1: float   # 25th percentile
    q3: float   # 75th percentile
    min_val: float
    max_val: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "min": self.min_val,
            "max": self.max_val,
            "ci_95": [self.ci_lower, self.ci_upper]
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for experiment results.

    Usage:
        analyzer = StatisticalAnalyzer()
        analyzer.load_data(csv_path)
        results = analyzer.analyze_all()
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

    def load_data(self, path: str) -> 'StatisticalAnalyzer':
        """Load data from CSV file"""
        if not SCIPY_AVAILABLE:
            raise ImportError("pandas and scipy required for analysis")

        self.data = pd.read_csv(path)
        return self

    def load_dataframe(self, df: 'pd.DataFrame') -> 'StatisticalAnalyzer':
        """Load data from DataFrame"""
        self.data = df
        return self

    # =========================================================================
    # Descriptive Statistics
    # =========================================================================

    def descriptive_stats(self, column: str, group_by: str = None) -> Dict:
        """
        Compute descriptive statistics.

        Args:
            column: Column to analyze
            group_by: Optional grouping column

        Returns:
            Dict of DescriptiveStats (one per group if grouped)
        """
        if self.data is None:
            raise ValueError("No data loaded")

        def compute_stats(values) -> DescriptiveStats:
            values = values.dropna()
            n = len(values)
            if n == 0:
                return None

            mean = values.mean()
            std = values.std()
            sem = std / math.sqrt(n) if n > 0 else 0

            # 95% CI using t-distribution
            if n > 1:
                t_crit = stats.t.ppf(0.975, n - 1)
                ci_margin = t_crit * sem
            else:
                ci_margin = 0

            return DescriptiveStats(
                n=n,
                mean=mean,
                std=std,
                sem=sem,
                median=values.median(),
                q1=values.quantile(0.25),
                q3=values.quantile(0.75),
                min_val=values.min(),
                max_val=values.max(),
                ci_lower=mean - ci_margin,
                ci_upper=mean + ci_margin
            )

        if group_by:
            results = {}
            for group, group_data in self.data.groupby(group_by):
                results[group] = compute_stats(group_data[column])
            return results
        else:
            return {"all": compute_stats(self.data[column])}

    # =========================================================================
    # Effect Size Calculations
    # =========================================================================

    @staticmethod
    def cohens_d(group1, group2) -> float:
        """
        Compute Cohen's d effect size.

        Interpretation:
        - Small: |d| < 0.2
        - Medium: 0.2 <= |d| < 0.8
        - Large: |d| >= 0.8
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()

        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (group1.mean() - group2.mean()) / pooled_std

    @staticmethod
    def eta_squared(ss_effect: float, ss_total: float) -> float:
        """
        Compute eta-squared effect size for ANOVA.

        Interpretation:
        - Small: η² < 0.01
        - Medium: 0.01 <= η² < 0.06
        - Large: η² >= 0.14
        """
        if ss_total == 0:
            return 0.0
        return ss_effect / ss_total

    @staticmethod
    def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
        """Compute partial eta-squared for factorial ANOVA"""
        if ss_effect + ss_error == 0:
            return 0.0
        return ss_effect / (ss_effect + ss_error)

    @staticmethod
    def interpret_effect_size(value: float, effect_type: str) -> str:
        """Interpret effect size magnitude"""
        if effect_type in ["d", "cohens_d"]:
            if abs(value) < 0.2:
                return "negligible"
            elif abs(value) < 0.5:
                return "small"
            elif abs(value) < 0.8:
                return "medium"
            else:
                return "large"
        elif effect_type in ["eta_squared", "partial_eta_squared"]:
            if value < 0.01:
                return "negligible"
            elif value < 0.06:
                return "small"
            elif value < 0.14:
                return "medium"
            else:
                return "large"
        return "unknown"

    # =========================================================================
    # Assumption Tests
    # =========================================================================

    def test_normality(self, column: str, group_by: str = None) -> Dict[str, StatisticalResult]:
        """
        Test normality using Shapiro-Wilk test.

        Note: For n > 5000, use Anderson-Darling instead.
        """
        results = {}

        if group_by:
            groups = self.data.groupby(group_by)[column]
        else:
            groups = [("all", self.data[column])]

        for name, values in groups:
            values = values.dropna()
            if len(values) < 3:
                continue

            if len(values) > 5000:
                # Use first 5000 for Shapiro-Wilk
                values = values.sample(5000, random_state=42)

            stat, p = shapiro(values)

            results[str(name)] = StatisticalResult(
                test_name="Shapiro-Wilk",
                statistic=stat,
                p_value=p,
                significant=p < 0.05,
                interpretation="Data is normally distributed" if p >= 0.05 else "Data is NOT normally distributed"
            )

        return results

    def test_homogeneity(self, column: str, group_by: str) -> StatisticalResult:
        """
        Test homogeneity of variance using Levene's test.
        """
        groups = [group[column].dropna().values for name, group in self.data.groupby(group_by)]

        stat, p = levene(*groups)

        return StatisticalResult(
            test_name="Levene's Test",
            statistic=stat,
            p_value=p,
            significant=p < 0.05,
            interpretation="Variances are homogeneous" if p >= 0.05 else "Variances are NOT homogeneous"
        )

    # =========================================================================
    # Parametric Tests
    # =========================================================================

    def independent_ttest(self, column: str, group_by: str,
                          groups: Tuple[str, str] = None) -> StatisticalResult:
        """
        Independent samples t-test.

        Args:
            column: Dependent variable
            group_by: Grouping variable
            groups: Specific groups to compare (optional)
        """
        grouped = self.data.groupby(group_by)[column]
        group_names = list(grouped.groups.keys())

        if groups:
            g1_name, g2_name = groups
        else:
            if len(group_names) != 2:
                raise ValueError(f"Expected 2 groups, got {len(group_names)}")
            g1_name, g2_name = group_names

        g1 = grouped.get_group(g1_name).dropna()
        g2 = grouped.get_group(g2_name).dropna()

        stat, p = ttest_ind(g1, g2)
        d = self.cohens_d(g1, g2)

        return StatisticalResult(
            test_name=f"Independent t-test ({g1_name} vs {g2_name})",
            statistic=stat,
            p_value=p,
            effect_size=d,
            effect_size_type="Cohen's d",
            df=len(g1) + len(g2) - 2,
            significant=p < 0.05,
            interpretation=self.interpret_effect_size(d, "d")
        )

    def one_way_anova(self, column: str, group_by: str) -> StatisticalResult:
        """
        One-way ANOVA.

        Tests if there are significant differences between group means.
        """
        groups = [group[column].dropna().values for name, group in self.data.groupby(group_by)]

        stat, p = f_oneway(*groups)

        # Compute eta-squared
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)

        eta_sq = self.eta_squared(ss_between, ss_total)

        df_between = len(groups) - 1
        df_within = len(all_data) - len(groups)

        return StatisticalResult(
            test_name="One-way ANOVA",
            statistic=stat,
            p_value=p,
            effect_size=eta_sq,
            effect_size_type="eta_squared",
            df=(df_between, df_within),
            significant=p < 0.05,
            interpretation=self.interpret_effect_size(eta_sq, "eta_squared")
        )

    def two_way_anova(self, dv: str, factor1: str, factor2: str) -> Dict[str, StatisticalResult]:
        """
        Two-way ANOVA.

        Tests main effects and interaction.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for two-way ANOVA")

        formula = f'{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'
        model = ols(formula, data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        results = {}

        for idx, row in anova_table.iterrows():
            if 'Residual' in str(idx):
                continue

            name = str(idx).replace('C(', '').replace(')', '').replace(':', ' × ')
            ss_effect = row['sum_sq']
            ss_total = anova_table['sum_sq'].sum()
            eta_sq = self.eta_squared(ss_effect, ss_total)

            results[name] = StatisticalResult(
                test_name=f"Two-way ANOVA: {name}",
                statistic=row['F'],
                p_value=row['PR(>F)'],
                effect_size=eta_sq,
                effect_size_type="eta_squared",
                df=(row['df'], anova_table.loc['Residual', 'df']),
                significant=row['PR(>F)'] < 0.05,
                interpretation=self.interpret_effect_size(eta_sq, "eta_squared")
            )

        return results

    def post_hoc_tukey(self, column: str, group_by: str) -> 'pd.DataFrame':
        """
        Tukey HSD post-hoc test.

        Use after significant ANOVA to find which groups differ.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Tukey HSD")

        tukey = pairwise_tukeyhsd(
            self.data[column].dropna(),
            self.data[group_by].dropna()
        )

        return pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )

    # =========================================================================
    # Non-parametric Tests
    # =========================================================================

    def kruskal_wallis(self, column: str, group_by: str) -> StatisticalResult:
        """
        Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).
        """
        groups = [group[column].dropna().values for name, group in self.data.groupby(group_by)]

        stat, p = kruskal(*groups)

        # Effect size: epsilon-squared
        n = sum(len(g) for g in groups)
        epsilon_sq = (stat - len(groups) + 1) / (n - len(groups))

        return StatisticalResult(
            test_name="Kruskal-Wallis H-test",
            statistic=stat,
            p_value=p,
            effect_size=epsilon_sq,
            effect_size_type="epsilon_squared",
            significant=p < 0.05
        )

    def mann_whitney_u(self, column: str, group_by: str,
                       groups: Tuple[str, str] = None) -> StatisticalResult:
        """
        Mann-Whitney U test (non-parametric alternative to independent t-test).
        """
        grouped = self.data.groupby(group_by)[column]
        group_names = list(grouped.groups.keys())

        if groups:
            g1_name, g2_name = groups
        else:
            if len(group_names) != 2:
                raise ValueError(f"Expected 2 groups, got {len(group_names)}")
            g1_name, g2_name = group_names

        g1 = grouped.get_group(g1_name).dropna()
        g2 = grouped.get_group(g2_name).dropna()

        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')

        # Effect size: rank-biserial correlation
        n1, n2 = len(g1), len(g2)
        r = 1 - (2 * stat) / (n1 * n2)

        return StatisticalResult(
            test_name=f"Mann-Whitney U ({g1_name} vs {g2_name})",
            statistic=stat,
            p_value=p,
            effect_size=r,
            effect_size_type="rank_biserial_r",
            significant=p < 0.05
        )

    # =========================================================================
    # Categorical Tests
    # =========================================================================

    def chi_square_test(self, var1: str, var2: str) -> StatisticalResult:
        """
        Chi-square test of independence.
        """
        contingency = pd.crosstab(self.data[var1], self.data[var2])
        chi2, p, dof, expected = chi2_contingency(contingency)

        # Cramer's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        return StatisticalResult(
            test_name=f"Chi-square ({var1} × {var2})",
            statistic=chi2,
            p_value=p,
            effect_size=cramers_v,
            effect_size_type="Cramer's V",
            df=dof,
            significant=p < 0.05,
            additional_info={"expected": expected.tolist()}
        )

    # =========================================================================
    # Comprehensive Analysis
    # =========================================================================

    def analyze_safety_by_method(self) -> Dict:
        """
        Analyze safety metrics (is_correct, violated) by method.
        """
        results = {}

        # Accuracy by method
        accuracy_by_method = self.data.groupby('method')['is_correct'].mean()
        results['accuracy_by_method'] = accuracy_by_method.to_dict()

        # Chi-square test: method × is_correct
        results['chi_square'] = self.chi_square_test('method', 'is_correct').to_dict()

        # If more than 2 methods, do post-hoc pairwise comparisons
        methods = self.data['method'].unique()
        if len(methods) > 2:
            pairwise = {}
            for i, m1 in enumerate(methods):
                for m2 in methods[i+1:]:
                    subset = self.data[self.data['method'].isin([m1, m2])]
                    if len(subset) > 0:
                        contingency = pd.crosstab(subset['method'], subset['is_correct'])
                        if contingency.shape == (2, 2):
                            _, p = fisher_exact(contingency)
                            pairwise[f"{m1}_vs_{m2}"] = {"p_value": p, "significant": p < 0.05}
            results['pairwise_fisher'] = pairwise

        return results

    def analyze_latency_by_method(self) -> Dict:
        """
        Analyze decision latency by method.
        """
        results = {}

        # Descriptive stats
        results['descriptive'] = {
            k: v.to_dict() for k, v in
            self.descriptive_stats('decision_latency_ms', 'method').items()
            if v is not None
        }

        # Normality test
        normality = self.test_normality('decision_latency_ms', 'method')
        all_normal = all(not r.significant for r in normality.values())
        results['normality'] = {k: v.to_dict() for k, v in normality.items()}

        # Homogeneity test
        homogeneity = self.test_homogeneity('decision_latency_ms', 'method')
        results['homogeneity'] = homogeneity.to_dict()

        # Main test
        if all_normal and not homogeneity.significant:
            # Use parametric ANOVA
            anova = self.one_way_anova('decision_latency_ms', 'method')
            results['main_test'] = anova.to_dict()

            if anova.significant:
                # Post-hoc
                tukey = self.post_hoc_tukey('decision_latency_ms', 'method')
                results['post_hoc'] = tukey.to_dict()
        else:
            # Use non-parametric Kruskal-Wallis
            kw = self.kruskal_wallis('decision_latency_ms', 'method')
            results['main_test'] = kw.to_dict()

        return results

    def analyze_by_scenario(self) -> Dict:
        """
        Analyze results by scenario.
        """
        results = {}

        # Accuracy by scenario
        accuracy_by_scenario = self.data.groupby('scenario')['is_correct'].mean()
        results['accuracy_by_scenario'] = accuracy_by_scenario.to_dict()

        # Two-way ANOVA: method × scenario
        if STATSMODELS_AVAILABLE:
            try:
                two_way = self.two_way_anova('is_correct', 'method', 'scenario')
                results['two_way_anova'] = {k: v.to_dict() for k, v in two_way.items()}
            except Exception as e:
                results['two_way_anova_error'] = str(e)

        return results

    def analyze_all(self) -> Dict:
        """
        Run all standard analyses.

        Returns comprehensive analysis results.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        results = {
            'metadata': {
                'n_trials': len(self.data),
                'methods': self.data['method'].unique().tolist() if 'method' in self.data else [],
                'scenarios': self.data['scenario'].unique().tolist() if 'scenario' in self.data else [],
            }
        }

        # Safety analysis
        if 'is_correct' in self.data.columns and 'method' in self.data.columns:
            results['safety_by_method'] = self.analyze_safety_by_method()

        # Latency analysis
        if 'decision_latency_ms' in self.data.columns and 'method' in self.data.columns:
            results['latency_by_method'] = self.analyze_latency_by_method()

        # Scenario analysis
        if 'scenario' in self.data.columns:
            results['by_scenario'] = self.analyze_by_scenario()

        self.results = results
        return results

    def save_results(self, path: str):
        """Save analysis results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def generate_latex_table(self, analysis_type: str = 'safety') -> str:
        """
        Generate LaTeX table for paper.

        Args:
            analysis_type: 'safety', 'latency', or 'summary'

        Returns:
            LaTeX table string
        """
        if analysis_type == 'safety':
            return self._latex_safety_table()
        elif analysis_type == 'latency':
            return self._latex_latency_table()
        else:
            return self._latex_summary_table()

    def _latex_safety_table(self) -> str:
        """Generate LaTeX table for safety results"""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Safety Performance by Method}",
            r"\label{tab:safety}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & Accuracy (\%) & Violations & Effect Size \\",
            r"\midrule"
        ]

        if 'safety_by_method' in self.results:
            for method, acc in self.results['safety_by_method'].get('accuracy_by_method', {}).items():
                lines.append(f"{method} & {acc*100:.1f} & - & - \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        return '\n'.join(lines)

    def _latex_latency_table(self) -> str:
        """Generate LaTeX table for latency results"""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Decision Latency by Method}",
            r"\label{tab:latency}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Mean (ms) & SD & Median & 95\% CI \\",
            r"\midrule"
        ]

        if 'latency_by_method' in self.results:
            for method, stats in self.results['latency_by_method'].get('descriptive', {}).items():
                ci = stats.get('ci_95', [0, 0])
                lines.append(
                    f"{method} & {stats['mean']:.1f} & {stats['std']:.1f} & "
                    f"{stats['median']:.1f} & [{ci[0]:.1f}, {ci[1]:.1f}] \\\\"
                )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        return '\n'.join(lines)

    def _latex_summary_table(self) -> str:
        """Generate summary LaTeX table"""
        return "% Summary table - implement based on your needs"


def quick_analysis(csv_path: str) -> Dict:
    """
    Quick analysis function for command-line use.

    Args:
        csv_path: Path to results CSV

    Returns:
        Analysis results
    """
    analyzer = StatisticalAnalyzer()
    analyzer.load_data(csv_path)
    results = analyzer.analyze_all()

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*60)

    if 'safety_by_method' in results:
        print("\nSafety by Method:")
        for method, acc in results['safety_by_method'].get('accuracy_by_method', {}).items():
            print(f"  {method}: {acc*100:.1f}%")

    if 'latency_by_method' in results:
        print("\nLatency by Method:")
        for method, stats in results['latency_by_method'].get('descriptive', {}).items():
            print(f"  {method}: M={stats['mean']:.1f}ms, SD={stats['std']:.1f}")

        main_test = results['latency_by_method'].get('main_test', {})
        if main_test:
            print(f"\n  Main test: p={main_test.get('p_value', 'N/A'):.4f}")

    return results
