"""Copula visualization tools."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List
import seaborn as sns

from .base import BaseCopula


class CopulaVisualizer:
    """Visualization tools for copula analysis."""

    @staticmethod
    def plot_pdf_surface(
        copula: BaseCopula,
        title: Optional[str] = None,
        grid_size: int = 50,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 3D surface of copula PDF.

        Args:
            copula: Fitted copula instance
            title: Plot title
            grid_size: Number of grid points per dimension
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        # Compute PDF on grid
        Z = np.zeros_like(U)
        for i in range(grid_size):
            for j in range(grid_size):
                uv = np.array([[U[i, j], V[i, j]]])
                Z[i, j] = copula.pdf(uv)[0]

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(
            U, V, Z,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True,
            alpha=0.9
        )

        # Labels and title
        ax.set_xlabel('U₁', fontsize=12)
        ax.set_ylabel('U₂', fontsize=12)
        ax.set_zlabel('Density', fontsize=12)

        if title is None:
            title = f'{copula.__class__.__name__} PDF'
        ax.set_title(title, fontsize=14, pad=20)

        # Color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Viewing angle
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_cdf_contours(
        copula: BaseCopula,
        title: Optional[str] = None,
        grid_size: int = 100,
        levels: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot contour plot of copula CDF.

        Args:
            copula: Fitted copula instance
            title: Plot title
            grid_size: Number of grid points per dimension
            levels: Number of contour levels
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        # Compute CDF on grid
        Z = np.zeros_like(U)
        for i in range(grid_size):
            for j in range(grid_size):
                uv = np.array([[U[i, j], V[i, j]]])
                Z[i, j] = copula.cdf(uv)[0]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot contours
        contourf = ax.contourf(U, V, Z, levels=levels, cmap='viridis', alpha=0.8)
        contour = ax.contour(U, V, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.4)

        # Labels and title
        ax.set_xlabel('U₁', fontsize=12)
        ax.set_ylabel('U₂', fontsize=12)

        if title is None:
            title = f'{copula.__class__.__name__} CDF Contours'
        ax.set_title(title, fontsize=14)

        # Color bar
        cbar = fig.colorbar(contourf, ax=ax)
        cbar.set_label('C(u₁, u₂)', fontsize=12)

        # Add diagonal line for reference
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1, label='Independence')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_scatter_comparison(
        copula: BaseCopula,
        data: np.ndarray,
        title: Optional[str] = None,
        n_samples: int = 1000,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot scatter comparison between empirical data and copula samples.

        Args:
            copula: Fitted copula instance
            data: Empirical data (n_samples, 2)
            title: Plot title
            n_samples: Number of samples to generate from copula
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Generate samples from copula
        copula_samples = copula.sample(n_samples)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot empirical data
        ax1.scatter(data[:, 0], data[:, 1], alpha=0.5, s=20, c='blue', edgecolors='none')
        ax1.set_xlabel('U₁', fontsize=12)
        ax1.set_ylabel('U₂', fontsize=12)
        ax1.set_title('Empirical Data', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Plot copula samples
        ax2.scatter(copula_samples[:, 0], copula_samples[:, 1], alpha=0.5, s=20, c='red', edgecolors='none')
        ax2.set_xlabel('U₁', fontsize=12)
        ax2.set_ylabel('U₂', fontsize=12)
        ax2.set_title(f'{copula.__class__.__name__} Samples', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        # Overall title
        if title is None:
            title = 'Empirical vs Copula Comparison'
        fig.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_copula_comparison(
        copulas: List[BaseCopula],
        copula_names: List[str],
        grid_size: int = 50,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of multiple copulas.

        Args:
            copulas: List of fitted copula instances
            copula_names: Names for each copula
            grid_size: Number of grid points per dimension
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        n_copulas = len(copulas)
        fig, axes = plt.subplots(2, n_copulas, figsize=figsize)

        if n_copulas == 1:
            axes = axes.reshape(2, 1)

        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        for idx, (copula, name) in enumerate(zip(copulas, copula_names)):
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

            # Plot PDF contours
            ax_pdf = axes[0, idx]
            contourf_pdf = ax_pdf.contourf(U, V, Z_pdf, levels=15, cmap='viridis')
            ax_pdf.set_xlabel('U₁', fontsize=10)
            ax_pdf.set_ylabel('U₂', fontsize=10)
            ax_pdf.set_title(f'{name} PDF', fontsize=11, fontweight='bold')
            fig.colorbar(contourf_pdf, ax=ax_pdf, fraction=0.046, pad=0.04)

            # Plot CDF contours
            ax_cdf = axes[1, idx]
            contourf_cdf = ax_cdf.contourf(U, V, Z_cdf, levels=15, cmap='viridis')
            ax_cdf.set_xlabel('U₁', fontsize=10)
            ax_cdf.set_ylabel('U₂', fontsize=10)
            ax_cdf.set_title(f'{name} CDF', fontsize=11, fontweight='bold')
            fig.colorbar(contourf_cdf, ax=ax_cdf, fraction=0.046, pad=0.04)

        fig.suptitle('Copula Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_tail_dependence_illustration(
        copula: BaseCopula,
        data: np.ndarray,
        threshold: float = 0.1,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Illustrate tail dependence in copula and empirical data.

        Args:
            copula: Fitted copula instance
            data: Empirical data (n_samples, 2)
            threshold: Threshold for tail regions
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Lower tail region
        lower_tail = (data[:, 0] <= threshold) & (data[:, 1] <= threshold)
        ax1.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, c='gray', label='All data')
        ax1.scatter(data[lower_tail, 0], data[lower_tail, 1], alpha=0.8, s=20, c='red', label='Lower tail')
        ax1.axvline(threshold, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(threshold, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('U₁', fontsize=12)
        ax1.set_ylabel('U₂', fontsize=12)
        ax1.set_title('Lower Tail Dependence', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Upper tail region
        upper_threshold = 1 - threshold
        upper_tail = (data[:, 0] >= upper_threshold) & (data[:, 1] >= upper_threshold)
        ax2.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, c='gray', label='All data')
        ax2.scatter(data[upper_tail, 0], data[upper_tail, 1], alpha=0.8, s=20, c='blue', label='Upper tail')
        ax2.axvline(upper_threshold, color='blue', linestyle='--', alpha=0.5)
        ax2.axhline(upper_threshold, color='blue', linestyle='--', alpha=0.5)
        ax2.set_xlabel('U₁', fontsize=12)
        ax2.set_ylabel('U₂', fontsize=12)
        ax2.set_title('Upper Tail Dependence', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        # Calculate tail dependence coefficients
        if hasattr(copula, 'tail_dependence'):
            lambda_l, lambda_u = copula.tail_dependence()
            fig.suptitle(
                f'{copula.__class__.__name__}: λₗ = {lambda_l:.4f}, λᵤ = {lambda_u:.4f}',
                fontsize=14,
                fontweight='bold',
                y=1.02
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_density_heatmap(
        copula: BaseCopula,
        title: Optional[str] = None,
        grid_size: int = 100,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap of copula density.

        Args:
            copula: Fitted copula instance
            title: Plot title
            grid_size: Number of grid points per dimension
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        # Create grid
        u = np.linspace(0.01, 0.99, grid_size)
        v = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u, v)

        # Compute PDF on grid
        Z = np.zeros_like(U)
        for i in range(grid_size):
            for j in range(grid_size):
                uv = np.array([[U[i, j], V[i, j]]])
                Z[i, j] = copula.pdf(uv)[0]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(
            Z,
            extent=[0, 1, 0, 1],
            origin='lower',
            cmap='viridis',
            aspect='auto',
            interpolation='bilinear'
        )

        # Labels and title
        ax.set_xlabel('U₁', fontsize=12)
        ax.set_ylabel('U₂', fontsize=12)

        if title is None:
            title = f'{copula.__class__.__name__} Density Heatmap'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Density', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
