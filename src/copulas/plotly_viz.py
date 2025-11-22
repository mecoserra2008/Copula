"""Advanced interactive copula visualizations using Plotly (pure numpy backend)."""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .base import BaseCopula


class InteractiveCopulaVisualizer:
    """
    Advanced interactive visualization tools for copula analysis using Plotly.

    This class provides rich, interactive visualizations with:
    - 3D surface plots with rotation and zoom
    - Interactive contour plots
    - Animated parameter evolution
    - Manifold visualizations
    - Information geometry plots
    """

    def __init__(self):
        """Initialize the visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive visualizations. "
                "Install it with: pip install plotly"
            )

    @staticmethod
    def plot_3d_surface(
        copula: BaseCopula,
        mode: str = "pdf",
        title: Optional[str] = None,
        grid_size: int = 50,
        colorscale: str = "Viridis",
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive 3D surface plot of copula PDF or CDF.

        Args:
            copula: Fitted copula instance
            mode: "pdf" or "cdf"
            title: Plot title
            grid_size: Number of grid points per dimension
            colorscale: Plotly colorscale name
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        # Compute values on grid
        Z = np.zeros_like(U)
        for i in range(grid_size):
            for j in range(grid_size):
                uv = np.array([[U[i, j], V[i, j]]])
                if mode == "pdf":
                    Z[i, j] = copula.pdf(uv)[0]
                elif mode == "cdf":
                    Z[i, j] = copula.cdf(uv)[0]
                else:
                    raise ValueError(f"Invalid mode: {mode}")

        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=U,
            y=V,
            z=Z,
            colorscale=colorscale,
            colorbar=dict(title=mode.upper(), len=0.7),
            hovertemplate='u₁: %{x:.3f}<br>u₂: %{y:.3f}<br>' + f'{mode.upper()}: %{z:.4f}<extra></extra>',
        )])

        # Update layout
        if title is None:
            title = f'{copula.__class__.__name__} {mode.upper()} Surface'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            scene=dict(
                xaxis=dict(title='u₁', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                yaxis=dict(title='u₂', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                zaxis=dict(title=mode.upper(), backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig

    @staticmethod
    def plot_contour_interactive(
        copula: BaseCopula,
        mode: str = "pdf",
        title: Optional[str] = None,
        grid_size: int = 100,
        n_contours: int = 20,
        colorscale: str = "Viridis",
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive contour plot with hover information.

        Args:
            copula: Fitted copula instance
            mode: "pdf" or "cdf"
            title: Plot title
            grid_size: Number of grid points per dimension
            n_contours: Number of contour levels
            colorscale: Plotly colorscale name
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        # Compute values
        Z = np.zeros_like(U)
        for i in range(grid_size):
            for j in range(grid_size):
                uv = np.array([[U[i, j], V[i, j]]])
                if mode == "pdf":
                    Z[i, j] = copula.pdf(uv)[0]
                else:
                    Z[i, j] = copula.cdf(uv)[0]

        # Create contour plot
        fig = go.Figure(data=go.Contour(
            x=u,
            y=v,
            z=Z,
            colorscale=colorscale,
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white'),
            ),
            colorbar=dict(title=mode.upper(), len=0.8),
            hovertemplate='u₁: %{x:.3f}<br>u₂: %{y:.3f}<br>' + f'{mode.upper()}: %{z:.4f}<extra></extra>',
            ncontours=n_contours,
        ))

        # Add independence line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Independence',
            hoverinfo='name',
        ))

        if title is None:
            title = f'{copula.__class__.__name__} {mode.upper()} Contours'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis=dict(title='u₁', range=[0, 1]),
            yaxis=dict(title='u₂', range=[0, 1], scaleanchor="x", scaleratio=1),
            width=800,
            height=800,
            hovermode='closest',
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig

    @staticmethod
    def plot_scatter_comparison(
        copula: BaseCopula,
        empirical_data: np.ndarray,
        n_samples: int = 1000,
        title: Optional[str] = None,
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive scatter plot comparing empirical data with copula samples.

        Args:
            copula: Fitted copula instance
            empirical_data: Empirical uniform data (n, 2)
            n_samples: Number of copula samples to generate
            title: Plot title
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        # Generate copula samples
        copula_samples = copula.sample(n_samples)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Empirical Data', f'{copula.__class__.__name__} Samples'),
            horizontal_spacing=0.12,
        )

        # Empirical data scatter
        fig.add_trace(
            go.Scattergl(
                x=empirical_data[:, 0],
                y=empirical_data[:, 1],
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.5),
                name='Empirical',
                hovertemplate='u₁: %{x:.3f}<br>u₂: %{y:.3f}<extra></extra>',
            ),
            row=1, col=1
        )

        # Copula samples scatter
        fig.add_trace(
            go.Scattergl(
                x=copula_samples[:, 0],
                y=copula_samples[:, 1],
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.5),
                name='Copula',
                hovertemplate='u₁: %{x:.3f}<br>u₂: %{y:.3f}<extra></extra>',
            ),
            row=1, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="u₁", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="u₂", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="u₁", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="u₂", range=[0, 1], row=1, col=2)

        if title is None:
            title = 'Empirical vs Copula Samples Comparison'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            showlegend=True,
            width=1400,
            height=600,
            hovermode='closest',
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig

    @staticmethod
    def plot_conditional_distribution(
        copula: BaseCopula,
        v_values: List[float] = [0.25, 0.5, 0.75],
        title: Optional[str] = None,
        grid_size: int = 200,
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot conditional distribution C(u|v) for different v values.

        Args:
            copula: Fitted copula instance
            v_values: List of conditioning values
            title: Plot title
            grid_size: Number of grid points
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        u_grid = np.linspace(0.01, 0.99, grid_size)

        fig = go.Figure()

        for v in v_values:
            v_array = np.full_like(u_grid, v)
            cond_cdf = copula.conditional_cdf(u_grid, v_array, condition_on=1)

            fig.add_trace(go.Scatter(
                x=u_grid,
                y=cond_cdf,
                mode='lines',
                name=f'C(u|v={v})',
                line=dict(width=3),
                hovertemplate=f'u: %{{x:.3f}}<br>C(u|v={v}): %{{y:.3f}}<extra></extra>',
            ))

        # Add independence reference
        fig.add_trace(go.Scatter(
            x=u_grid,
            y=u_grid,
            mode='lines',
            name='Independence',
            line=dict(color='black', dash='dash', width=2),
            hovertemplate='u: %{x:.3f}<br>Independence: %{y:.3f}<extra></extra>',
        ))

        if title is None:
            title = f'{copula.__class__.__name__} Conditional Distribution'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis=dict(title='u', range=[0, 1]),
            yaxis=dict(title='C(u|v)', range=[0, 1]),
            width=900,
            height=600,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig

    @staticmethod
    def plot_tail_dependence_analysis(
        copula: BaseCopula,
        empirical_data: np.ndarray,
        title: Optional[str] = None,
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Analyze and visualize tail dependence structure.

        Args:
            copula: Fitted copula instance
            empirical_data: Empirical uniform data
            title: Plot title
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Lower Tail (u₁,u₂ < 0.1)', 'Upper Tail (u₁,u₂ > 0.9)'),
        )

        # Lower tail analysis
        lower_threshold = 0.1
        lower_mask = (empirical_data[:, 0] <= lower_threshold) & (empirical_data[:, 1] <= lower_threshold)

        # All data in gray
        fig.add_trace(
            go.Scatter(
                x=empirical_data[:, 0],
                y=empirical_data[:, 1],
                mode='markers',
                marker=dict(size=3, color='lightgray', opacity=0.3),
                name='All data',
                showlegend=True,
                hoverinfo='skip',
            ),
            row=1, col=1
        )

        # Lower tail in red
        fig.add_trace(
            go.Scatter(
                x=empirical_data[lower_mask, 0],
                y=empirical_data[lower_mask, 1],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.8),
                name='Lower tail',
                hovertemplate='u₁: %{x:.3f}<br>u₂: %{y:.3f}<extra></extra>',
            ),
            row=1, col=1
        )

        # Threshold lines
        fig.add_hline(y=lower_threshold, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        fig.add_vline(x=lower_threshold, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

        # Upper tail analysis
        upper_threshold = 0.9
        upper_mask = (empirical_data[:, 0] >= upper_threshold) & (empirical_data[:, 1] >= upper_threshold)

        # All data in gray
        fig.add_trace(
            go.Scatter(
                x=empirical_data[:, 0],
                y=empirical_data[:, 1],
                mode='markers',
                marker=dict(size=3, color='lightgray', opacity=0.3),
                name='All data',
                showlegend=False,
                hoverinfo='skip',
            ),
            row=1, col=2
        )

        # Upper tail in blue
        fig.add_trace(
            go.Scatter(
                x=empirical_data[upper_mask, 0],
                y=empirical_data[upper_mask, 1],
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.8),
                name='Upper tail',
                hovertemplate='u₁: %{x:.3f}<br>u₂: %{y:.3f}<extra></extra>',
            ),
            row=1, col=2
        )

        # Threshold lines
        fig.add_hline(y=upper_threshold, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=2)
        fig.add_vline(x=upper_threshold, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=2)

        # Update axes
        fig.update_xaxes(title_text="u₁", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="u₂", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="u₁", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="u₂", range=[0, 1], row=1, col=2)

        # Get tail dependence coefficients
        if hasattr(copula, 'tail_dependence'):
            lambda_l, lambda_u = copula.tail_dependence()
            if title is None:
                title = f'{copula.__class__.__name__}: λₗ={lambda_l:.4f}, λᵤ={lambda_u:.4f}'
        else:
            if title is None:
                title = 'Tail Dependence Analysis'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            width=1400,
            height=600,
            hovermode='closest',
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig

    @staticmethod
    def plot_copula_comparison_grid(
        copulas: List[BaseCopula],
        names: List[str],
        grid_size: int = 50,
        title: Optional[str] = None,
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Create grid comparison of multiple copulas.

        Args:
            copulas: List of fitted copula instances
            names: Names for each copula
            grid_size: Number of grid points
            title: Plot title
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        n_copulas = len(copulas)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=n_copulas,
            subplot_titles=[f'{name} PDF' for name in names] + [f'{name} CDF' for name in names],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        for idx, (copula, name) in enumerate(zip(copulas, names)):
            # Compute PDF
            Z_pdf = np.zeros_like(U)
            for i in range(grid_size):
                for j in range(grid_size):
                    uv = np.array([[U[i, j], V[i, j]]])
                    Z_pdf[i, j] = copula.pdf(uv)[0]

            # Compute CDF
            Z_cdf = np.zeros_like(U)
            for i in range(grid_size):
                for j in range(grid_size):
                    uv = np.array([[U[i, j], V[i, j]]])
                    Z_cdf[i, j] = copula.cdf(uv)[0]

            # Add PDF contour
            fig.add_trace(
                go.Contour(
                    x=u, y=v, z=Z_pdf,
                    colorscale='Viridis',
                    showscale=(idx == n_copulas - 1),
                    colorbar=dict(title='PDF', x=1.02) if idx == n_copulas - 1 else None,
                    hovertemplate=f'{name}<br>u₁: %{{x:.3f}}<br>u₂: %{{y:.3f}}<br>PDF: %{{z:.4f}}<extra></extra>',
                ),
                row=1, col=idx + 1
            )

            # Add CDF contour
            fig.add_trace(
                go.Contour(
                    x=u, y=v, z=Z_cdf,
                    colorscale='Plasma',
                    showscale=(idx == n_copulas - 1),
                    colorbar=dict(title='CDF', x=1.02) if idx == n_copulas - 1 else None,
                    hovertemplate=f'{name}<br>u₁: %{{x:.3f}}<br>u₂: %{{y:.3f}}<br>CDF: %{{z:.4f}}<extra></extra>',
                ),
                row=2, col=idx + 1
            )

            # Update axes
            fig.update_xaxes(title_text="u₁", row=1, col=idx + 1)
            fig.update_yaxes(title_text="u₂", row=1, col=idx + 1)
            fig.update_xaxes(title_text="u₁", row=2, col=idx + 1)
            fig.update_yaxes(title_text="u₂", row=2, col=idx + 1)

        if title is None:
            title = 'Copula Comparison Grid'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
            width=500 * n_copulas,
            height=900,
            showlegend=False,
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig

    @staticmethod
    def plot_parameter_sensitivity(
        copula_class,
        param_range: Dict[str, np.ndarray],
        U_test: np.ndarray,
        title: Optional[str] = None,
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> go.Figure:
        """
        Visualize how copula density changes with parameter values.

        Args:
            copula_class: Copula class to instantiate
            param_range: Dictionary mapping parameter names to arrays of values
            U_test: Test points (n_points, 2)
            title: Plot title
            show: Whether to display the plot
            save_html: Path to save HTML file

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Get parameter name and values
        param_name = list(param_range.keys())[0]
        param_values = param_range[param_name]

        for param_val in param_values:
            # Create and fit copula with this parameter
            cop = copula_class()

            # Manually set parameter for visualization
            if hasattr(cop, 'params_'):
                cop.params_[param_name] = param_val
                cop.is_fitted_ = True
            else:
                cop.params_ = {param_name: param_val}
                cop.is_fitted_ = True
                cop.n_obs_ = 100

            # Compute PDF
            pdf_values = cop.pdf(U_test)

            fig.add_trace(go.Scatter(
                y=pdf_values,
                mode='lines+markers',
                name=f'{param_name}={param_val:.3f}',
                line=dict(width=2),
                marker=dict(size=4),
                hovertemplate=f'{param_name}={param_val:.3f}<br>Point: %{{x}}<br>PDF: %{{y:.4f}}<extra></extra>',
            ))

        if title is None:
            title = f'Parameter Sensitivity Analysis: {param_name}'

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis=dict(title='Test Point Index'),
            yaxis=dict(title='PDF Value'),
            width=1000,
            height=600,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
        )

        if save_html:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig
