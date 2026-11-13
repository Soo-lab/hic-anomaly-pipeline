cli needs (viz): --report (call write_html_report(outdir, run_info, figures, calls_df)); --triangular (plot_heatmap triangular=True); --no-log-heatmap (log=False).
Pipeline should call plot_heatmap(matrix, chrom, bin_size, out, calls=regions), plot_score_track(scores, thr, bin_size, out, calls=regions).
