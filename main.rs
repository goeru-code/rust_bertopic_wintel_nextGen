use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use eframe::egui;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

const ZINC_950: egui::Color32 = egui::Color32::from_rgb(9, 9, 11);
const ZINC_900: egui::Color32 = egui::Color32::from_rgb(24, 24, 27);
const ZINC_800: egui::Color32 = egui::Color32::from_rgb(39, 39, 42);
const ZINC_700: egui::Color32 = egui::Color32::from_rgb(63, 63, 70);
const ZINC_600: egui::Color32 = egui::Color32::from_rgb(82, 82, 91);
const ZINC_400: egui::Color32 = egui::Color32::from_rgb(161, 161, 170);
const ZINC_100: egui::Color32 = egui::Color32::from_rgb(244, 244, 245);
const SKY_100: egui::Color32 = egui::Color32::from_rgb(224, 242, 254);
const SKY_250: egui::Color32 = egui::Color32::from_rgb(186, 230, 253);
const SKY_400: egui::Color32 = egui::Color32::from_rgb(56, 189, 248);
const SKY_550: egui::Color32 = egui::Color32::from_rgb(14, 165, 233);
const SKY_700: egui::Color32 = egui::Color32::from_rgb(3, 105, 161);

// Topic, record ID, resolution word count.
type TopicMinimum = (i64, String, usize);

fn apply_dark_zinc_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    visuals.panel_fill = ZINC_950;
    visuals.window_fill = ZINC_950;
    visuals.faint_bg_color = ZINC_900;
    visuals.extreme_bg_color = ZINC_900;

    visuals.widgets.noninteractive.bg_fill = ZINC_900;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, ZINC_800);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, ZINC_100);

    visuals.widgets.inactive.bg_fill = ZINC_800;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, ZINC_700);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, ZINC_100);

    visuals.widgets.hovered.bg_fill = ZINC_700;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ZINC_600);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, ZINC_100);

    visuals.widgets.active.bg_fill = ZINC_600;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ZINC_400);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, ZINC_100);

    visuals.selection.bg_fill = ZINC_700;
    visuals.selection.stroke = egui::Stroke::new(1.0, ZINC_100);

    ctx.set_visuals(visuals);
}

fn output_wrapped_cell(ui: &mut egui::Ui, width: f32, value: &str, is_header: bool) {
    ui.vertical(|ui| {
        ui.set_width(width);

        let rich_text = if is_header {
            egui::RichText::new(value)
                .monospace()
                .strong()
                .color(ZINC_100)
        } else {
            egui::RichText::new(value).monospace().color(ZINC_100)
        };

        ui.add(egui::Label::new(rich_text).wrap(true));
    });
}

fn main() -> eframe::Result<()> {
    pyo3::prepare_freethreaded_python();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1750.0, 1000.0])
            .with_title("BERTopic Rust - paraphrase-multilingual-MiniLM-L12-v2"),
        ..Default::default()
    };

    eframe::run_native(
        "BERTopic Rust",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            apply_dark_zinc_theme(&cc.egui_ctx);
            Box::new(BerTopicApp::default())
        }),
    )
}

enum PipelineMessage {
    Success(String, Vec<TopicMinimum>),
    Error(String),
}

enum RestAuthMessage {
    Success(String),
    Error(String),
}

enum RestFetchMessage {
    Success(Vec<String>),
    Error(String),
}

struct BerTopicApp {
    // Remedy REST configuration and state
    rest_base_url: String,
    rest_login_path: String,
    rest_entry_path: String,
    rest_assigned_group: String,
    rest_days_back: u64,
    rest_username: String,
    rest_password: String,
    rest_status: String,
    rest_token: Option<String>,
    rest_auth_rx: Option<Receiver<RestAuthMessage>>,
    rest_auth_pending: bool,
    rest_fetch_rx: Option<Receiver<RestFetchMessage>>,
    rest_fetch_pending: bool,

    input_text: String,
    output_file: String,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    cluster_selection_method: String,
    status_message: String,
    is_processing: bool,
    output_text: String,
    summary_items: Vec<String>,
    topic_resolution_minima: Vec<TopicMinimum>,
    input_record_count: usize,
    rx: Option<Receiver<PipelineMessage>>,
    pending_scroll_topic: Option<String>,
    pending_scroll_record_id: Option<String>,
    show_text: bool,
    show_optional_column: bool,
    text_column_width: f32,
    optional_column_width: f32,
}

impl Default for BerTopicApp {
    fn default() -> Self {
        Self {
            rest_base_url: "https://smartoneitsm-itg-restapi.bmc-itg.corproot.net".to_string(),
            rest_login_path: "/api/jwt/login".to_string(),
            rest_entry_path: "/api/arsys/v1/entry/HPD:Help Desk".to_string(),
            rest_assigned_group: "ESM-ITSM%".to_string(),
            rest_days_back: 30,
            rest_username: String::new(),
            rest_password: String::new(),
            rest_status: "Enter credentials, then test the login.".to_string(),
            rest_token: None,
            rest_auth_rx: None,
            rest_auth_pending: false,
            rest_fetch_rx: None,
            rest_fetch_pending: false,

            input_text: String::new(),
            output_file: "output.txt".to_string(),
            n_neighbors: 25,
            min_cluster_size: 15,
            min_samples: 5,
            cluster_selection_method: "eom".to_string(),
            status_message: "Ready.".to_string(),
            is_processing: false,
            output_text: String::new(),
            summary_items: Vec::new(),
            topic_resolution_minima: Vec::new(),
            input_record_count: 0,
            rx: None,
            pending_scroll_topic: None,
            pending_scroll_record_id: None,
            show_text: true,
            show_optional_column: true,
            text_column_width: 500.0,
            optional_column_width: 500.0,
        }
    }
}

impl BerTopicApp {
    fn build_summary(&mut self) -> String {
        let mut seen = HashSet::new();
        let mut summary_lines = Vec::new();

        for line in self.output_text.lines() {
            if line.starts_with("INPUT_RECORDS:")
                || line.starts_with("topic_score")
                || line.trim().is_empty()
            {
                continue;
            }

            let parts: Vec<&str> = line.split('|').map(str::trim).collect();

            if parts.len() >= 3 {
                let key = format!("topic {} share {}", parts[1], parts[2]);

                if seen.insert(key.clone()) {
                    summary_lines.push(key);
                }
            }
        }

        self.summary_items = summary_lines.clone();

        let mut out = String::new();
        out.push_str("\n\n================================================\n");
        out.push_str(&format!(
            "Summarized Analysis ({} input records)\n",
            self.input_record_count
        ));
        out.push_str("================================================\n");

        for item in &summary_lines {
            out.push_str(item);
            out.push('\n');
        }

        out
    }

    fn start_rest_login(&mut self, ctx: egui::Context) {
        let base_url = self.rest_base_url.trim().trim_end_matches('/');
        let login_path = self.rest_login_path.trim();
        let username = self.rest_username.trim().to_string();
        let password = self.rest_password.clone();

        if base_url.is_empty() || login_path.is_empty() {
            self.rest_status = "Enter the Remedy base URL and login path.".to_string();
            return;
        }

        if !base_url.starts_with("https://") {
            self.rest_status = "Use an HTTPS base URL.".to_string();
            return;
        }

        if username.is_empty() || password.is_empty() {
            self.rest_status = "Enter your Remedy username and password.".to_string();
            return;
        }

        let login_url = if login_path.starts_with('/') {
            format!("{base_url}{login_path}")
        } else {
            format!("{base_url}/{login_path}")
        };

        self.rest_token = None;
        self.rest_auth_pending = true;
        self.rest_status = "Logging in to Remedy...".to_string();

        let (tx, rx) = channel::<RestAuthMessage>();
        self.rest_auth_rx = Some(rx);

        thread::spawn(move || {
            let result = (|| -> Result<String, String> {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(30))
                    .build()
                    .map_err(|_| "Could not create the HTTPS client.".to_string())?;

                let response = client
                    .post(&login_url)
                    .form(&[
                        ("username", username.as_str()),
                        ("password", password.as_str()),
                    ])
                    .send()
                    .map_err(|_| "Could not connect to the Remedy login endpoint.".to_string())?;

                let status = response.status();
                if !status.is_success() {
                    return Err(format!("Remedy login failed (HTTP {status})."));
                }

                let body = response.text().map_err(|_| {
                    "Login succeeded, but the token response could not be read.".to_string()
                })?;

                let token = body
                    .trim()
                    .strip_prefix("AR-JWT ")
                    .unwrap_or(body.trim())
                    .to_string();

                if token.is_empty() {
                    return Err("Login response was successful but contained no token.".to_string());
                }

                Ok(token)
            })();

            let message = match result {
                Ok(token) => RestAuthMessage::Success(token),
                Err(error) => RestAuthMessage::Error(error),
            };

            let _ = tx.send(message);
            ctx.request_repaint();
        });
    }

    fn check_rest_auth_jobs(&mut self) {
        let received = match self.rest_auth_rx.as_ref() {
            Some(rx) => match rx.try_recv() {
                Ok(message) => Some(message),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(RestAuthMessage::Error(
                    "Login task ended without returning a result.".to_string(),
                )),
            },
            None => None,
        };

        if let Some(message) = received {
            match message {
                RestAuthMessage::Success(token) => {
                    self.rest_token = Some(token);
                    self.rest_password.clear();
                    self.rest_status =
                        "Login successful. Token is stored in memory for this session.".to_string();
                }
                RestAuthMessage::Error(message) => {
                    self.rest_token = None;
                    self.rest_status = message;
                }
            }

            self.rest_auth_pending = false;
            self.rest_auth_rx = None;
        }
    }

    fn start_rest_fetch(&mut self, ctx: egui::Context) {
        // Clear old results immediately whenever Fetch Records is clicked.
        self.input_text.clear();
        self.input_record_count = 0;

        let Some(token) = self.rest_token.clone() else {
            self.rest_status = "Test the Remedy login before fetching records.".to_string();
            return;
        };

        let base_url = self.rest_base_url.trim().trim_end_matches('/').to_string();
        let entry_path = self.rest_entry_path.trim().to_string();
        let assigned_group = self.rest_assigned_group.trim().to_string();

        if !base_url.starts_with("https://") {
            self.rest_status = "Use an HTTPS base URL.".to_string();
            return;
        }

        if entry_path.starts_with("http://") || entry_path.starts_with("https://") {
            self.rest_status =
                "Entry path must be a path on the configured Remedy server.".to_string();
            return;
        }

        if assigned_group.is_empty() {
            self.rest_status = "Enter an Assigned Group filter, e.g. ESM-ITSM%.".to_string();
            return;
        }

        if assigned_group.contains('"')
            || assigned_group.contains('\n')
            || assigned_group.contains('\r')
        {
            self.rest_status =
                "Assigned Group filter must not contain quotes or line breaks.".to_string();
            return;
        }

        let entry_path = if entry_path.starts_with('/') {
            entry_path
        } else {
            format!("/{entry_path}")
        };

        let days_back = self.rest_days_back.max(1);

        self.rest_fetch_pending = true;
        self.rest_status = "Fetching Remedy records...".to_string();

        let (tx, rx) = channel::<RestFetchMessage>();
        self.rest_fetch_rx = Some(rx);

        thread::spawn(move || {
            let result =
                fetch_incident_records(base_url, entry_path, token, days_back, assigned_group);

            let message = match result {
                Ok(records) => RestFetchMessage::Success(records),
                Err(error) => RestFetchMessage::Error(error),
            };

            let _ = tx.send(message);
            ctx.request_repaint();
        });
    }

    fn check_rest_fetch_jobs(&mut self) {
        let received = match self.rest_fetch_rx.as_ref() {
            Some(rx) => match rx.try_recv() {
                Ok(message) => Some(message),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(RestFetchMessage::Error(
                    "Fetch task ended without returning a result.".to_string(),
                )),
            },
            None => None,
        };

        if let Some(message) = received {
            match message {
                RestFetchMessage::Success(records) => {
                    self.input_record_count = records.len();
                    self.input_text = records.join("\n");
                    self.rest_status = format!(
                        "Retrieved {} records with ID, description, and resolution.",
                        records.len()
                    );
                }
                RestFetchMessage::Error(error) => {
                    self.rest_status = error;
                }
            }

            self.rest_fetch_pending = false;
            self.rest_fetch_rx = None;
        }
    }

    fn run_pipeline(&mut self, ctx: egui::Context) {
        if self.input_text.trim().is_empty() {
            self.status_message = "Error: Input text area is empty.".to_string();
            return;
        }

        let requested_path = self.output_file.trim();
        if requested_path.is_empty() {
            self.status_message = "Error: Output file path is empty.".to_string();
            return;
        }

        let path = PathBuf::from(requested_path);
        let output_path = if path.is_absolute() {
            path
        } else {
            match std::env::current_dir() {
                Ok(current_dir) => current_dir.join(path),
                Err(e) => {
                    self.status_message = format!("Could not resolve output path: {e}");
                    return;
                }
            }
        };

        self.output_file = output_path.to_string_lossy().into_owned();

        self.input_record_count = self
            .input_text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();

        self.status_message = "Processing topics... Please wait.".to_string();
        self.is_processing = true;

        let (tx, rx): (Sender<PipelineMessage>, Receiver<PipelineMessage>) = channel();
        self.rx = Some(rx);

        let input_content = self.input_text.clone();
        let n_neighbors = self.n_neighbors;
        let min_cluster_size = self.min_cluster_size;
        let min_samples = self.min_samples;
        let selection_method = self.cluster_selection_method.clone();

        thread::spawn(move || {
            let result = py_result(
                input_content,
                n_neighbors,
                min_cluster_size,
                min_samples,
                selection_method,
            );

            match result {
                Ok((out, topic_minima)) => {
                    if out.starts_with("Error:") {
                        let _ = tx.send(PipelineMessage::Error(out));
                    } else {
                        let _ = tx.send(PipelineMessage::Success(out, topic_minima));
                    }
                }
                Err(e) => {
                    let _ = tx.send(PipelineMessage::Error(format!("Execution Error: {}", e)));
                }
            }

            ctx.request_repaint();
        });
    }

    fn check_background_jobs(&mut self) {
        if let Some(rx) = &self.rx {
            if let Ok(msg) = rx.try_recv() {
                match msg {
                    PipelineMessage::Success(out, topic_minima) => {
                        self.output_text = out;
                        self.topic_resolution_minima = topic_minima;

                        let summary_section = self.build_summary();
                        let final_output = format!("{}{}", self.output_text, summary_section);

                        self.output_text = final_output.clone();

                        match fs::write(&self.output_file, &final_output) {
                            Ok(()) => {
                                self.status_message = format!(
                                    "Analysis complete. Output written to {}",
                                    self.output_file
                                );
                            }
                            Err(e) => {
                                self.status_message = format!(
                                    "Analysis completed, but could not write {}: {}",
                                    self.output_file, e
                                );
                            }
                        }
                    }
                    PipelineMessage::Error(err_msg) => {
                        self.status_message = err_msg;
                    }
                }

                self.is_processing = false;
                self.rx = None;
            }
        }
    }
}

fn fetch_incident_records(
    base_url: String,
    entry_path: String,
    token: String,
    days_back: u64,
    assigned_group: String,
) -> Result<Vec<String>, String> {
    const PAGE_LIMIT: usize = 500;
    const MAX_RECORDS: usize = 50_000;
    const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

    let assigned_group = assigned_group.trim();

    if assigned_group.contains('"')
        || assigned_group.contains('\n')
        || assigned_group.contains('\r')
    {
        return Err("Assigned Group filter must not contain quotes or line breaks.".to_string());
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| "Could not determine the current time.".to_string())?
        .as_secs();

    let cutoff = now.saturating_sub(days_back.saturating_mul(SECONDS_PER_DAY));

    let qualification =
        format!("('Submit Date' > \"{cutoff}\") AND ('Assigned Group' LIKE \"{assigned_group}\")");

    let url = format!("{base_url}{entry_path}");

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|_| "Could not create the HTTPS client.".to_string())?;

    let mut records = Vec::new();
    let mut seen = HashSet::new();
    let mut offset = 0usize;

    loop {
        let limit_string = PAGE_LIMIT.to_string();
        let offset_string = offset.to_string();

        let response = client
            .get(&url)
            .header("Authorization", format!("AR-JWT {token}"))
            .query(&[
                ("q", qualification.as_str()),
                (
                    "fields",
                    "values(Incident Number,Description,Detailed Decription,Resolution)",
                ),
                ("limit", limit_string.as_str()),
                ("offset", offset_string.as_str()),
            ])
            .send()
            .map_err(|_| "Could not connect to the Remedy entry endpoint.".to_string())?;

        let status = response.status();
        if !status.is_success() {
            return Err(format!("Remedy record request failed (HTTP {status})."));
        }

        let body = response
            .text()
            .map_err(|_| "Could not read the Remedy response.".to_string())?;

        let payload: serde_json::Value = serde_json::from_str(&body)
            .map_err(|_| "Remedy returned a response that was not valid JSON.".to_string())?;

        let entries = payload
            .get("entries")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "Response did not contain an entries array.".to_string())?;

        let received = entries.len();
        if received == 0 {
            break;
        }

        for entry in entries {
            let values = entry.get("values");

            let incident_number = values
                .and_then(|v| v.get("Incident Number"))
                .map(json_value_to_text)
                .unwrap_or_default();

            if incident_number.is_empty() || !seen.insert(incident_number.clone()) {
                continue;
            }

            let description = values
                .and_then(|v| v.get("Description"))
                .map(json_value_to_text)
                .unwrap_or_default();

            let detailed_description = values
                .and_then(|v| v.get("Detailed Decription"))
                .map(json_value_to_text)
                .unwrap_or_default();

            let resolution = values
                .and_then(|v| v.get("Resolution"))
                .map(json_value_to_text)
                .unwrap_or_default();

            let combined_description = format!(
                "{} - {}",
                single_line(&description),
                single_line(&detailed_description)
            );

            let record_line = format!(
                "{}\t{}\t{}",
                quote_tsv_field(&incident_number),
                quote_tsv_field(&combined_description),
                quote_tsv_field(&single_line(&resolution)),
            );

            records.push(record_line);

            if records.len() >= MAX_RECORDS {
                return Err(format!(
                    "Reached the safety limit of {MAX_RECORDS} records. Reduce the days-back range."
                ));
            }
        }

        offset = offset.saturating_add(received);
    }

    Ok(records)
}

fn json_value_to_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.trim().to_string(),
        serde_json::Value::Number(number) => number.to_string(),
        _ => String::new(),
    }
}

fn single_line(value: &str) -> String {
    value
        .replace('\t', " ")
        .replace('\r', " ")
        .replace('\n', " ")
        .trim()
        .to_string()
}

fn quote_tsv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn py_result(
    input_content: String,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    selection_method: String,
) -> PyResult<(String, Vec<TopicMinimum>)> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        path.call_method1(
            "append",
            ("C:\\Users\\taaergo1\\AppData\\Local\\Programs\\Python\\Python314",),
        )?;

        let os = py.import("os")?;
        os.getattr("environ")?.set_item("HF_HUB_OFFLINE", "1")?;
        os.getattr("environ")?
            .set_item("TRANSFORMERS_OFFLINE", "1")?;

        let code = cr#"
import csv
import pandas as pd
import re
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

def execute(raw_text, n_neighbors, min_cluster_size, min_samples, method):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not lines:
        return "Error: Input text contains no valid lines.", []

    records = []

    for line in lines:
        try:
            parts = next(csv.reader(
                [line],
                delimiter="\t",
                quotechar='"',
                doublequote=True
            ))
        except (csv.Error, StopIteration):
            parts = []

        if len(parts) >= 3:
            rec_id = parts[0].strip()
            text_val = parts[1].strip()
            resolution_val = parts[2].strip()
        elif len(parts) == 2:
            rec_id = parts[0].strip()
            text_val = parts[1].strip()
            resolution_val = ""
        else:
            # Retain support for the older pasted format.
            match = re.match(r'^\s*(\S+)\s+"(.*?)"(?:\s+(.+?))?\s*$', line)

            if match:
                rec_id = match.group(1).strip()
                text_val = match.group(2).strip()
                resolution_val = (match.group(3) or "").strip()
            else:
                parts = re.split(r'\t+|\s+', line, maxsplit=1)

                if len(parts) == 2:
                    rec_id = parts[0].strip()
                    text_val = parts[1].strip().strip('"')
                else:
                    rec_id = "UNKNOWN"
                    text_val = line.strip().strip('"')

                resolution_val = ""

        records.append({
            "record_id": rec_id,
            "text": text_val,
            "resolution": resolution_val
        })

    df = pd.DataFrame(records)

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_model = SentenceTransformer(MODEL_NAME)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=method,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False,
    )

    # Topics are calculated from descriptions; resolutions are metadata.
    clean_texts = df["text"].tolist()
    topics, probs = topic_model.fit_transform(clean_texts)
    df["topic"] = topics

    # Count Unicode letter/number sequences; punctuation is ignored.
    def count_words(value):
        return len(re.findall(r"[^\W_]+", str(value), flags=re.UNICODE))

    df["resolution_word_count"] = df["resolution"].map(count_words)

    # Keep the first record when multiple records tie for the minimum.
    topic_minima = []
    for topic, topic_df in df.groupby("topic", sort=True):
        min_index = topic_df["resolution_word_count"].idxmin()
        row = df.loc[min_index]
        topic_minima.append((
            int(topic),
            str(row["record_id"]),
            int(row["resolution_word_count"])
        ))

    if probs is not None:
        df["topic_score"] = (
            probs.max(axis=1) if len(probs.shape) > 1 else probs
        )
    else:
        df["topic_score"] = 0.0

    topic_counts = df["topic"].value_counts()
    total_rows = len(df)

    topic_share_map = {
        topic: f"{(count / total_rows) * 100:.1f}%"
        for topic, count in topic_counts.items()
    }

    df["share"] = df["topic"].map(topic_share_map)

    ranked_df = df.sort_values(
        by=["topic", "topic_score"],
        ascending=[True, False]
    )

    output_columns = [
        "topic_score",
        "topic",
        "share",
        "record_id",
        "text",
        "resolution"
    ]

    export_df = ranked_df[output_columns].copy()

    export_df["text"] = export_df["text"].map(
        lambda x: x if len(x) <= 1000 else x[:997] + "..."
    )

    export_df["topic_score"] = export_df["topic_score"].map(
        lambda x: f"{x:.4f}"
    )

    def clean_field(value):
        return (
            str(value)
            .replace("\t", " ")
            .replace("\r", " ")
            .replace("\n", " ")
            .replace("|", "/")
        )

    header = (
        f"INPUT_RECORDS: {total_rows}\n\n"
        "topic_score | topic | share | record_id | text | resolution\n"
    )

    out_lines = [header]

    for _, row in export_df.iterrows():
        out_lines.append(
            f"{clean_field(row['topic_score'])} | "
            f"{clean_field(row['topic'])} | "
            f"{clean_field(row['share'])} | "
            f"{clean_field(row['record_id'])} | "
            f"{clean_field(row['text'])} | "
            f"{clean_field(row['resolution'])}\n"
        )

    return "".join(out_lines), topic_minima
"#;

        let module = PyModule::from_code(py, code, c"pipeline.py", c"pipeline")?;
        let func = module.getattr("execute")?;

        let result: (String, Vec<TopicMinimum>) = func
            .call1((
                input_content,
                n_neighbors,
                min_cluster_size,
                min_samples,
                selection_method,
            ))?
            .extract()?;

        Ok(result)
    })
}

impl eframe::App for BerTopicApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_background_jobs();
        self.check_rest_auth_jobs();
        self.check_rest_fetch_jobs();

        egui::TopBottomPanel::bottom("summary_panel")
            .resizable(true)
            .default_height(280.0)
            .min_height(150.0)
            .max_height(450.0)
            .show(ctx, |ui| {
                let panel_size = ui.available_size();
                let divider_spacing = ui.spacing().item_spacing.x;
                let left_width = (panel_size.x * 0.64)
                    .min(panel_size.x - divider_spacing - 200.0)
                    .max(0.0);
                let panel_height = panel_size.y;

                ui.horizontal(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(left_width, panel_height),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            egui::Frame::group(ui.style())
                                .fill(ZINC_800)
                                .stroke(egui::Stroke::new(1.0, ZINC_700))
                                .rounding(egui::Rounding::same(6.0))
                                .inner_margin(egui::Margin::same(10.0))
                                .show(ui, |ui| {
                                    ui.set_min_size(ui.available_size());

                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new("Summarized Analysis")
                                                .strong()
                                                .color(ZINC_100),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "({} input records)",
                                                self.input_record_count
                                            ))
                                            .color(ZINC_400),
                                        );
                                    });

                                    ui.add_space(8.0);

                                    egui::ScrollArea::vertical()
                                        .id_source("summary_scroll_area")
                                        .auto_shrink([false, false])
                                        .show(ui, |ui| {
                                            let spacing = 8.0;
                                            let button_width = ((ui.available_width()
                                                - spacing * 4.0)
                                                / 5.0)
                                                .min(160.0)
                                                .max(80.0);

                                            ui.set_width(ui.available_width());

                                            egui::Grid::new("summary_grid")
                                                .spacing(egui::vec2(spacing, spacing))
                                                .show(ui, |ui| {
                                                    for (idx, item) in
                                                        self.summary_items.iter().enumerate()
                                                    {
                                                        let bg = if idx % 2 == 0 {
                                                            ZINC_700
                                                        } else {
                                                            ZINC_600
                                                        };

                                                        let button = egui::Button::new(
                                                            egui::RichText::new(item)
                                                                .color(ZINC_100)
                                                                .strong(),
                                                        )
                                                        .wrap(true)
                                                        .fill(bg)
                                                        .stroke(egui::Stroke::new(1.0, ZINC_400))
                                                        .rounding(egui::Rounding::same(8.0));

                                                        if ui
                                                            .add_sized(
                                                                [button_width, 40.0],
                                                                button,
                                                            )
                                                            .clicked()
                                                        {
                                                            let parts: Vec<&str> =
                                                                item.split_whitespace().collect();

                                                            if parts.len() >= 2 {
                                                                self.pending_scroll_topic =
                                                                    Some(parts[1].to_string());
                                                                ctx.request_repaint();
                                                            }
                                                        }

                                                        if (idx + 1) % 5 == 0 {
                                                            ui.end_row();
                                                        }
                                                    }
                                                });
                                        });
                                });
                        },
                    );

                    ui.separator();

                    let right_width = ui.available_width();

                    ui.allocate_ui_with_layout(
                        egui::vec2(right_width, panel_height),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            egui::Frame::group(ui.style())
                                .fill(ZINC_800)
                                .stroke(egui::Stroke::new(1.0, ZINC_700))
                                .rounding(egui::Rounding::same(6.0))
                                .inner_margin(egui::Margin::same(10.0))
                                .show(ui, |ui| {
                                    ui.set_min_size(ui.available_size());

                                    ui.label(
                                        egui::RichText::new("Shortest resolution per topic")
                                            .strong()
                                            .color(ZINC_100),
                                    );
                                    ui.label(
                                        egui::RichText::new(
                                            "Punctuation is ignored when counting words.",
                                        )
                                        .color(ZINC_400),
                                    );
                                    ui.separator();

                                    egui::ScrollArea::vertical()
                                        .id_source("topic_resolution_minima")
                                        .auto_shrink([false, false])
                                        .show(ui, |ui| {
                                            for (topic, record_id, word_count) in
                                                &self.topic_resolution_minima
                                            {
                                                let label = format!(
                                                    "Topic {topic} — Record: {record_id} — {word_count} words"
                                                );

                                                if ui
                                                    .add(
                                                        egui::Button::new(label)
                                                            .wrap(true)
                                                            .fill(ZINC_700)
                                                            .stroke(egui::Stroke::new(
                                                                1.0, ZINC_600,
                                                            )),
                                                    )
                                                    .clicked()
                                                {
                                                    self.pending_scroll_record_id =
                                                        Some(record_id.clone());
                                                    ctx.request_repaint();
                                                }

                                                ui.separator();
                                            }
                                        });
                                });
                        },
                    );
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header and Info popup.
            egui::Frame::default()
                .fill(ZINC_900)
                .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("BERTopic Rust");

                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.menu_button("Info", |ui| {
                                    ui.set_min_width(800.0);

                                    egui::ScrollArea::vertical()
                                        .max_height(800.0)
                                        .show(ui, |ui| {
                                            ui.label("BERTopic Rust Tool");
                                            ui.separator();
                                            ui.label(
                                                "Data records analysis / topic building in offline mode.",
                                            );
                                            ui.label("For sensitive data, for example.");
                                            ui.add_space(8.0);

                                            ui.label(
                                                r#"-- MS SQL syntax
DECLARE @YearsBack int = 1;

SELECT
    incident_number,
    '"' + REPLACE(
        ISNULL(description, '') + ' ' + ISNULL(Detailed_Decription, ''),
        '"',
        '""'
    ) + '"' AS text,
    Resolution
FROM hpd_help_desk
WHERE submit_date > DATEDIFF_BIG(
    SECOND,
    '19700101',
    DATEADD(YEAR, -@YearsBack, GETUTCDATE())
)
AND assigned_group LIKE 'ESM-ITSM%';"#,
                                            );

                                            ui.add_space(12.0);

                                            ui.add(
                                                egui::Image::new(egui::include_image!("info.jpg"))
                                                    .fit_to_exact_size(egui::vec2(800.0, 450.0)),
                                            );
                                        });
                                });
                            },
                        );
                    });
                });

            ui.add_space(8.0);

            // Remedy REST settings and actions.
            egui::Frame::group(ui.style())
                .fill(ZINC_800)
                .stroke(egui::Stroke::new(1.0, ZINC_700))
                .rounding(egui::Rounding::same(6.0))
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());

                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("BMC Remedy REST Data Retrieval")
                                .strong()
                                .color(ZINC_100),
                        );

                        if ui
                            .add_enabled(
                                !self.rest_auth_pending,
                                egui::Button::new(if self.rest_auth_pending {
                                    "Logging in..."
                                } else {
                                    "Test Login"
                                }),
                            )
                            .clicked()
                        {
                            self.start_rest_login(ctx.clone());
                        }

                        if ui
                            .add_enabled(
                                self.rest_token.is_some() && !self.rest_fetch_pending,
                                egui::Button::new(if self.rest_fetch_pending {
                                    "Fetching records..."
                                } else {
                                    "Fetch Records"
                                }),
                            )
                            .clicked()
                        {
                            self.start_rest_fetch(ctx.clone());
                        }
                    });

                    let rest_status = if self.rest_token.is_some() {
                        format!("Login successful. {}", self.rest_status)
                    } else {
                        self.rest_status.clone()
                    };

                    ui.label(egui::RichText::new(rest_status).color(ZINC_400));
                    ui.label(
                        egui::RichText::new(
                            "Fetch retrieves ID, combined description, and resolution; clicking Fetch Records clears the input area first.",
                        )
                        .color(ZINC_400),
                    );

                    ui.add_space(8.0);

                    // REST fields on one line when the window has enough width.
                    ui.horizontal_wrapped(|ui| {
                        ui.vertical(|ui| {
                            ui.label("Base URL");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rest_base_url)
                                    .desired_width(400.0),
                            );
                        });

                        ui.vertical(|ui| {
                            ui.label("Login path");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rest_login_path)
                                    .desired_width(130.0),
                            );
                        });

                        ui.vertical(|ui| {
                            ui.label("Entry path");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rest_entry_path)
                                    .desired_width(210.0),
                            );
                        });

                        ui.vertical(|ui| {
                            ui.label("Username");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rest_username)
                                    .desired_width(77.0),
                            );
                        });

                        ui.vertical(|ui| {
                            ui.label("Password");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rest_password)
                                    .password(true)
                                    .desired_width(77.0),
                            );
                        });

                        ui.vertical(|ui| {
                            ui.label("Assigned Group (LIKE)");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rest_assigned_group)
                                    .hint_text("ESM-ITSM%")
                                    .desired_width(130.0),
                            );
                        });

                        ui.vertical(|ui| {
                            ui.label("Days back");
                            ui.scope(|ui| {
                                ui.spacing_mut().slider_width = 110.0;

                                let visuals = ui.visuals_mut();
                                visuals.widgets.inactive.bg_fill = ZINC_600;
                                visuals.widgets.hovered.bg_fill = ZINC_400;
                                visuals.widgets.active.bg_fill = ZINC_100;
                                visuals.selection.bg_fill = ZINC_400;

                                ui.add(
                                    egui::Slider::new(&mut self.rest_days_back, 1..=3650)
                                        .suffix(" days"),
                                );
                            });
                        });
                    });
                });

            ui.add_space(8.0);

            // Input section; four visible rows.
            egui::Frame::group(ui.style())
                .fill(ZINC_700)
                .stroke(egui::Stroke::new(1.0, ZINC_600))
                .rounding(egui::Rounding::same(6.0))
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.visuals_mut().extreme_bg_color = ZINC_700;
                    ui.set_width(ui.available_width());

                    ui.label(
                        egui::RichText::new(
                            "Input Text (record ID, quoted description, and resolution; one per line)",
                        )
                        .strong()
                        .color(ZINC_100),
                    );

                    ui.add_space(5.0);

                    egui::ScrollArea::vertical()
                        .id_source("input_scroll_area")
                        .max_height(70.0)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.input_text)
                                    .hint_text(
                                        "INC000014999999\t\"Description - Detailed Decription\"\t\"Resolution text\"",
                                    )
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(4),
                            );
                        });
                });

            ui.add_space(10.0);

            // Parameters, with labels beside their controls.
            egui::Frame::group(ui.style())
                .fill(ZINC_800)
                .stroke(egui::Stroke::new(1.0, ZINC_700))
                .rounding(egui::Rounding::same(6.0))
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        egui::RichText::new("Parameters")
                            .strong()
                            .color(ZINC_100),
                    );
                    ui.add_space(5.0);
                    ui.spacing_mut().slider_width = 180.0;

                    ui.horizontal_wrapped(|ui| {
                        egui::Frame::group(ui.style())
                            .fill(SKY_100)
                            .rounding(egui::Rounding::same(6.0))
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("Output File")
                                            .size(16.0)
                                            .color(ZINC_950),
                                    );
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.output_file)
                                            .desired_width(220.0),
                                    );
                                });
                            });

                        egui::Frame::group(ui.style())
                            .fill(SKY_250)
                            .rounding(egui::Rounding::same(6.0))
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("n_neighbors")
                                            .size(16.0)
                                            .color(ZINC_950),
                                    );
                                    ui.add_sized(
                                        [220.0, 0.0],
                                        egui::Slider::new(&mut self.n_neighbors, 2..=100),
                                    );
                                });
                            });

                        egui::Frame::group(ui.style())
                            .fill(SKY_400)
                            .rounding(egui::Rounding::same(6.0))
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("min_cluster_size")
                                            .size(16.0)
                                            .color(ZINC_100),
                                    );
                                    ui.add_sized(
                                        [220.0, 0.0],
                                        egui::Slider::new(
                                            &mut self.min_cluster_size,
                                            2..=50,
                                        ),
                                    );
                                });
                            });

                        egui::Frame::group(ui.style())
                            .fill(SKY_550)
                            .rounding(egui::Rounding::same(6.0))
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("min_samples")
                                            .size(16.0)
                                            .color(ZINC_100),
                                    );
                                    ui.add_sized(
                                        [220.0, 0.0],
                                        egui::Slider::new(&mut self.min_samples, 1..=30),
                                    );
                                });
                            });

                        egui::Frame::group(ui.style())
                            .fill(SKY_700)
                            .rounding(egui::Rounding::same(6.0))
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("Selection Method")
                                            .size(16.0)
                                            .color(ZINC_100),
                                    );

                                    egui::ComboBox::from_id_source("method_combo")
                                        .width(140.0)
                                        .selected_text(
                                            egui::RichText::new(
                                                &self.cluster_selection_method,
                                            )
                                            .size(16.0)
                                            .color(ZINC_100),
                                        )
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.cluster_selection_method,
                                                "eom".to_string(),
                                                egui::RichText::new("eom").size(16.0),
                                            );
                                            ui.selectable_value(
                                                &mut self.cluster_selection_method,
                                                "leaf".to_string(),
                                                egui::RichText::new("leaf").size(16.0),
                                            );
                                        });
                                });
                            });
                    });
                });

            ui.add_space(10.0);

            // Run button and status.
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !self.is_processing,
                        egui::Button::new("Run BERTopic Pipeline"),
                    )
                    .clicked()
                {
                    self.run_pipeline(ctx.clone());
                }

                ui.label(&self.status_message);
            });

            ui.add_space(10.0);

            // Pipeline output with zebra-striped rows.
            egui::Frame::group(ui.style())
                .fill(ZINC_900)
                .stroke(egui::Stroke::new(1.0, ZINC_800))
                .rounding(egui::Rounding::same(6.0))
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.visuals_mut().extreme_bg_color = ZINC_900;
                    ui.set_width(ui.available_width());

                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("BERTopic Pipeline Output")
                                .strong()
                                .color(ZINC_100),
                        );

                        if ui
                            .button(if self.show_text {
                                "Collapse text"
                            } else {
                                "Expand text"
                            })
                            .clicked()
                        {
                            self.show_text = !self.show_text;
                        }

                        if ui
                            .button(if self.show_optional_column {
                                "Collapse resolution"
                            } else {
                                "Expand resolution"
                            })
                            .clicked()
                        {
                            self.show_optional_column = !self.show_optional_column;
                        }

                        ui.add(
                            egui::Slider::new(
                                &mut self.text_column_width,
                                150.0..=1200.0,
                            )
                            .text("Text width"),
                        );

                        ui.add(
                            egui::Slider::new(
                                &mut self.optional_column_width,
                                150.0..=1200.0,
                            )
                            .text("Resolution width"),
                        );
                    });

                    ui.add_space(5.0);

                    let output_height = ui.available_height();

                    egui::ScrollArea::both()
                        .id_source("output_scroll_area")
                        .auto_shrink([false, false])
                        .max_height(output_height)
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());

                            let mut result_row_index = 0usize;

                            for line in self.output_text.lines() {
                                let parts: Vec<&str> =
                                    line.split('|').map(str::trim).collect();

                                if parts.len() == 6 {
                                    let is_header = parts[0] == "topic_score";

                                    let row_fill = if is_header {
                                        ZINC_700
                                    } else {
                                        let fill = if result_row_index % 2 == 0 {
                                            ZINC_900
                                        } else {
                                            ZINC_800
                                        };
                                        result_row_index += 1;
                                        fill
                                    };

                                    let topic_id = if is_header {
                                        None
                                    } else {
                                        Some(parts[1].to_string())
                                    };

                                    let row_response = egui::Frame::default()
                                        .fill(row_fill)
                                        .inner_margin(egui::Margin::symmetric(6.0, 4.0))
                                        .show(ui, |ui| {
                                            ui.set_min_width(ui.available_width());

                                            ui.horizontal(|ui| {
                                                for value in &parts[..4] {
                                                    let text = egui::RichText::new(*value)
                                                        .monospace()
                                                        .color(ZINC_100);

                                                    let text = if is_header {
                                                        text.strong()
                                                    } else {
                                                        text
                                                    };

                                                    ui.label(text);
                                                    ui.add_space(8.0);
                                                }

                                                if self.show_text {
                                                    output_wrapped_cell(
                                                        ui,
                                                        self.text_column_width,
                                                        parts[4],
                                                        is_header,
                                                    );
                                                }

                                                if self.show_optional_column {
                                                    output_wrapped_cell(
                                                        ui,
                                                        self.optional_column_width,
                                                        parts[5],
                                                        is_header,
                                                    );
                                                }
                                            });
                                        });

                                    if let Some(topic) = topic_id {
                                        if self.pending_scroll_topic.as_deref()
                                            == Some(topic.as_str())
                                        {
                                            ui.scroll_to_rect(
                                                row_response.response.rect,
                                                Some(egui::Align::TOP),
                                            );
                                            self.pending_scroll_topic = None;
                                        }
                                    }

                                    if self.pending_scroll_record_id.as_deref()
                                        == Some(parts[3])
                                    {
                                        ui.scroll_to_rect(
                                            row_response.response.rect,
                                            Some(egui::Align::TOP),
                                        );
                                        self.pending_scroll_record_id = None;
                                    }
                                } else {
                                    ui.label(
                                        egui::RichText::new(line)
                                            .monospace()
                                            .color(ZINC_100),
                                    );
                                }
                            }
                        });
                });
        });
    }
}
