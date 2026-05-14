# Build the US panel from WRDS.
#
# Pulls quarterly fundamentals from Compustat and delisting events from
# CRSP, builds a financial-distress event from the CRSP distress
# delisting codes (the closest US analogue to the Chinese
# Special-Treatment flag), computes 32 financial ratios, and reshapes to
# the wide MIDAS layout the training scripts expect.
#
# Output: period_final_us.csv, one row per firm, 32 covariates x 23
# quarterly lags plus 5 metadata columns.
#
# Requires a WRDS account with Compustat and CRSP access. Run locally;
# do not commit credentials to disk.

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(dbplyr)
  library(tidyr)
  library(lubridate)
  library(zoo)
  library(data.table)
})

START_DATE <- as.Date("1985-01-01")
END_DATE   <- as.Date("2020-12-31")
N_FIRMS_TARGET <- 8000L

LAG_USE_YEAR <- 6L
QUARTER      <- 4L
JMAX         <- LAG_USE_YEAR * QUARTER - 1L
K_VARS       <- 32L

if (!exists("OUT_DIR")) {
  OUT_DIR <- tryCatch(
    normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path),
                            "..", "data"), mustWork = FALSE),
    error = function(e) getwd())
  if (identical(OUT_DIR, "/../data") || OUT_DIR == "" || is.na(OUT_DIR)) OUT_DIR <- getwd()
}
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
if (!exists("OUT_CSV")) OUT_CSV <- file.path(OUT_DIR, "period_final_us.csv")
cat(sprintf("Output will be written to: %s\n", OUT_CSV))

if (!exists("WRDS_USER")) WRDS_USER <- rstudioapi::askForPassword("WRDS username")
if (!exists("WRDS_PWD"))  WRDS_PWD  <- rstudioapi::askForPassword("WRDS password")

wrds <- dbConnect(Postgres(),
                  host     = "wrds-pgdata.wharton.upenn.edu",
                  port     = 9737,
                  dbname   = "wrds",
                  sslmode  = "require",
                  user     = WRDS_USER,
                  password = WRDS_PWD)
cat("Connected to WRDS\n")

fundq_fields <- c(
  "gvkey","datadate","fyearq","fqtr","datafqtr","conm","cusip","tic",
  "indfmt","datafmt","popsrc","consol","curcdq",
  "atq","actq","cheq","rectq","invtq","ppentq","intanq",
  "ltq","lctq","dlcq","dlttq",
  "seqq","ceqq","req",
  "revtq","saleq","cogsq","oibdpq","oiadpq","niq","xintq","txtq",
  "oancfy","dpq",
  "prccq","cshoq","mkvaltq",
  "dvpsxq"
)

cat("Pulling comp.fundq -- this may take a few minutes...\n")

q_fundq <- sprintf(
  "SELECT %s FROM comp.fundq
   WHERE indfmt IN ('INDL','FS') AND datafmt='STD' AND popsrc='D' AND consol='C'
     AND datadate BETWEEN '%s' AND '%s'
     AND curcdq = 'USD'",
  paste(fundq_fields, collapse = ","), START_DATE, END_DATE)

fundq <- dbGetQuery(wrds, q_fundq) |> as.data.table()
cat(sprintf("  rows pulled: %d, unique gvkeys: %d\n",
            nrow(fundq), uniqueN(fundq$gvkey)))

setorder(fundq, gvkey, fyearq, fqtr)
fundq[, oancfq := ifelse(fqtr == 1L, oancfy,
                         oancfy - shift(oancfy, 1L)), by = .(gvkey, fyearq)]
fundq[, yq := as.yearqtr(datadate)]

cat("Pulling CRSP delisting...\n")

q_dlst <- "
SELECT d.permno, d.dlstdt, d.dlstcd
FROM crsp.dse d
WHERE d.event = 'DELIST'
  AND d.dlstcd BETWEEN 400 AND 599
"
dlst <- tryCatch(dbGetQuery(wrds, q_dlst),
                 error = function(e) {
                   message("crsp.dse query failed, falling back to msedelist")
                   dbGetQuery(wrds,
                     "SELECT permno, dlstdt, dlstcd FROM crsp.msedelist
                      WHERE dlstcd BETWEEN 400 AND 599")
                 }) |> as.data.table()

ccm_link <- dbGetQuery(wrds, "
  SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
  FROM crsp.ccmxpf_lnkhist
  WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
") |> as.data.table()
ccm_link[is.na(linkenddt), linkenddt := as.Date("2099-12-31")]

dlst <- merge(dlst, ccm_link, by = "permno", allow.cartesian = TRUE)
dlst <- dlst[dlstdt >= linkdt & dlstdt <= linkenddt,
             .(gvkey, distress_date = as.Date(dlstdt),
               distress_src = paste0("crsp_dlst_", dlstcd))]
dlst <- unique(dlst[order(gvkey, distress_date)], by = "gvkey")
cat(sprintf("  CRSP financial delistings: %d firms\n", nrow(dlst)))

distress_all <- dlst[!is.na(gvkey) & !is.na(distress_date)]
setorder(distress_all, gvkey, distress_date)
distress <- distress_all[, .SD[1L], by = gvkey]
cat(sprintf("Distress events (unique firms): %d\n", nrow(distress)))

setorder(fundq, gvkey, datadate)

roll4 <- function(x) frollsum(x, n = 4L, fill = NA_real_, align = "right")
sr <- function(num, den) ifelse(is.na(num) | is.na(den) | den == 0, NA_real_, num / den)

fundq[, `:=`(
  saleq_4q    = roll4(saleq),
  oibdpq_4q   = roll4(oibdpq),
  oiadpq_4q   = roll4(oiadpq),
  niq_4q      = roll4(niq),
  oancfq_4q   = roll4(oancfq),
  net_debt    = (dlcq + dlttq) - cheq,
  tang_assets = atq - intanq,
  mve         = prccq * cshoq
), by = gvkey]

fundq[, `:=`(
  v01_AR_turnover            = sr(saleq_4q, rectq),
  v02_CA_turnover            = sr(saleq_4q, actq),
  v03_CA_TA                  = sr(actq, atq),
  v04_CL_TL                  = sr(lctq, ltq),
  v05_Debt_Ratio             = sr(ltq, atq),
  v06_EBITDA_TL              = sr(oibdpq_4q, ltq),
  v07_EBIT_Sales             = sr(oiadpq_4q, saleq_4q),
  v08_Equity_Multiplier      = sr(atq, seqq),
  v09_Fixed_Asset_Turnover   = sr(saleq_4q, ppentq),
  v10_Interest_Bearing_Debt  = sr(dlcq + dlttq, atq),
  v11_BPS                    = sr(seqq, cshoq),
  v12_OCF_Sales              = sr(oancfq, saleq),
  v13_OCF_per_share          = sr(oancfq, cshoq),
  v14_OCF_OI                 = sr(oancfq_4q, saleq_4q),
  v15_Net_Profit_TA          = sr(niq_4q, atq),
  v16_Net_Margin             = sr(niq_4q, saleq_4q),
  v17_Pretax_Per_Div         = sr(niq + txtq, dvpsxq * cshoq),
  v18_ROA                    = sr(niq_4q, atq),
  v19_Tang_NetDebt           = sr(tang_assets, net_debt),
  v20_SE_TL                  = sr(seqq, ltq),
  v21_TangAsset_TL           = sr(tang_assets, ltq),
  v22_Z1_WC_TA               = sr(actq - lctq, atq),
  v23_Z2_RE_TA               = sr(req, atq),
  v24_Z3_EBIT_TA             = sr(oiadpq_4q, atq),
  v25_Z4_MVE_TL              = sr(mve, ltq),
  v26_Z5_Sales_TA            = sr(saleq_4q, atq),
  v27_CF_Debt                = sr(oancfq_4q, dlcq + dlttq),
  v28_Current_Ratio          = sr(actq, lctq),
  v29_Equity_Ratio           = sr(seqq, atq),
  v30_OI_per_share           = sr(oiadpq, cshoq),
  v31_Quick_Ratio            = sr(actq - invtq, lctq),
  v32_TA_Turnover            = sr(saleq_4q, atq)
)]

ratio_cols <- sprintf("v%02d_%s", 1:32, c(
  "AR_turnover","CA_turnover","CA_TA","CL_TL","Debt_Ratio",
  "EBITDA_TL","EBIT_Sales","Equity_Multiplier","Fixed_Asset_Turnover",
  "Interest_Bearing_Debt","BPS","OCF_Sales","OCF_per_share","OCF_OI",
  "Net_Profit_TA","Net_Margin","Pretax_Per_Div","ROA","Tang_NetDebt",
  "SE_TL","TangAsset_TL","Z1_WC_TA","Z2_RE_TA","Z3_EBIT_TA","Z4_MVE_TL",
  "Z5_Sales_TA","CF_Debt","Current_Ratio","Equity_Ratio","OI_per_share",
  "Quick_Ratio","TA_Turnover"))
stopifnot(length(ratio_cols) == K_VARS)

for (rc in ratio_cols) {
  q1 <- quantile(fundq[[rc]], 0.01, na.rm = TRUE)
  q99 <- quantile(fundq[[rc]], 0.99, na.rm = TRUE)
  fundq[[rc]] <- pmin(pmax(fundq[[rc]], q1), q99)
}

panel <- fundq[, c("gvkey","datadate","yq", ratio_cols), with = FALSE]
panel <- merge(panel, distress[, .(gvkey, distress_date, distress_src)],
               by = "gvkey", all.x = TRUE)

panel[, has_obs := rowSums(!is.na(.SD)) >= K_VARS / 2, .SDcols = ratio_cols]
panel <- panel[has_obs == TRUE]

setorder(panel, gvkey, datadate)
panel <- panel[is.na(distress_date) | datadate <= distress_date]

firm_n <- panel[, .N, by = gvkey][N >= JMAX]
panel <- panel[gvkey %in% firm_n$gvkey]
cat(sprintf("Firms with >= %d valid quarters: %d\n", JMAX, nrow(firm_n)))

panel[, rk := frank(-as.numeric(datadate), ties.method = "first"), by = gvkey]
panel_lag <- panel[rk <= JMAX]
panel_lag[, lag := JMAX + 1L - rk]

long <- melt(panel_lag, id.vars = c("gvkey","lag"),
             measure.vars = ratio_cols,
             variable.name = "var", value.name = "value")
long[, col := sprintf("%s__lag%02d", var, lag)]
wide_ratios <- dcast(long, gvkey ~ col, value.var = "value")

ordered_cols <- as.vector(t(outer(ratio_cols,
                                  sprintf("lag%02d", 1:JMAX),
                                  paste, sep = "__")))
setcolorder(wide_ratios, c("gvkey", ordered_cols))

start_per_firm <- panel_lag[lag == 1L, .(gvkey, start_day = datadate)]
end_per_firm   <- panel_lag[, .(end_day = max(datadate)), by = gvkey]
distress_per_firm <- distress[gvkey %in% wide_ratios$gvkey,
                              .(gvkey, distress_date)]

meta <- merge(start_per_firm, end_per_firm, by = "gvkey")
meta <- merge(meta, distress_per_firm, by = "gvkey", all.x = TRUE)
meta[, status := as.integer(!is.na(distress_date))]

fmt_us <- function(d) format(as.Date(d), "%m/%d/%Y")
meta[, bankrupt_day := ifelse(status == 1L, fmt_us(distress_date), fmt_us(END_DATE))]
meta[, survival_time := as.numeric(
  ifelse(status == 1L, as.numeric(distress_date - start_day),
         as.numeric(END_DATE - start_day))
) / 365]
meta[, censoringtime := as.numeric(END_DATE - start_day) / 365]
meta[, start_day := as.character(start_day)]

final <- merge(wide_ratios, meta[, .(gvkey, start_day, bankrupt_day,
                                     survival_time, status, censoringtime)],
               by = "gvkey")

cat(sprintf("Pre-cap: %d firms, %d events (%.1f%%)\n",
            nrow(final), sum(final$status), 100*mean(final$status)))

if (nrow(final) > N_FIRMS_TARGET) {
  set.seed(20260510)
  evt <- final[status == 1L]
  noevt <- final[status == 0L]
  n_keep_noevt <- N_FIRMS_TARGET - nrow(evt)
  if (n_keep_noevt < 0) {
    final <- evt[sample(.N, N_FIRMS_TARGET)]
  } else if (n_keep_noevt < nrow(noevt)) {
    final <- rbind(evt, noevt[sample(.N, n_keep_noevt)])
  }
}

cat(sprintf("Final: %d firms, %d events (%.1f%%)\n",
            nrow(final), sum(final$status), 100*mean(final$status)))

final[, gvkey := NULL]

fwrite(final, OUT_CSV, sep = ";", dec = ",", na = "", row.names = FALSE)
cat(sprintf("Wrote %s (%d cols, %d rows)\n", OUT_CSV, ncol(final), nrow(final)))

dbDisconnect(wrds)
cat("Done.\n")

