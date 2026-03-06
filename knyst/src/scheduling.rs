//! This module contains things related to scheduling that are more generic than
//! graph internals.

use std::sync::{Arc, RwLock};

use crate::time::{Beats, Seconds};

/// A change in musical tempo for use in a [`MusicalTimeMap`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-derive", derive(serde::Serialize, serde::Deserialize))]
pub enum TempoChange {
    /// New BPM value.
    NewTempo {
        /// Tempo in beats per minute.
        bpm: f64,
    },
}

impl TempoChange {
    /// Give the duration in seconds of the tempo change.
    pub fn to_secs_f64(&self, duration: Beats) -> f64 {
        match self {
            Self::NewTempo { bpm } => (duration.as_beats_f64() * 60.0) / bpm,
        }
    }

    /// Converts a duration in seconds within this tempo change to beats.
    pub fn secs_f64_to_beats(&self, section_duration: f64) -> Beats {
        match self {
            Self::NewTempo { bpm } => Beats::from_beats_f64(*bpm * (section_duration / 60.0)),
        }
    }
}

/// Tempo-curve validation error.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum TempoCurveError {
    /// One tempo-curve field is invalid.
    #[error("invalid tempo curve: {0}")]
    Invalid(String),
}

/// One curved tempo segment relative to the previous tempo point.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-derive", derive(serde::Serialize, serde::Deserialize))]
pub struct TempoCurveSegment {
    duration: Seconds,
    target_bpm: f64,
    curve: f32,
}

impl TempoCurveSegment {
    /// Creates one tempo-curve segment.
    pub fn new(duration: Seconds, target_bpm: f64, curve: f32) -> Result<Self, TempoCurveError> {
        validate_segment(duration, target_bpm, curve)?;
        Ok(Self {
            duration,
            target_bpm,
            curve,
        })
    }

    /// Returns the segment duration.
    pub fn duration(&self) -> Seconds {
        self.duration
    }

    /// Returns the segment target tempo in BPM.
    pub fn target_bpm(&self) -> f64 {
        self.target_bpm
    }

    /// Returns the segment curvature.
    pub fn curve(&self) -> f32 {
        self.curve
    }
}

/// One full continuous transport tempo description anchored in seconds.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-derive", derive(serde::Serialize, serde::Deserialize))]
pub struct TempoCurve {
    start_position: Seconds,
    start_bpm: f64,
    segments: Vec<TempoCurveSegment>,
}

impl TempoCurve {
    /// Creates one tempo curve.
    pub fn new(
        start_position: Seconds,
        start_bpm: f64,
        segments: Vec<TempoCurveSegment>,
    ) -> Result<Self, TempoCurveError> {
        validate_start(start_position, start_bpm)?;
        Ok(Self {
            start_position,
            start_bpm,
            segments,
        })
    }

    /// Returns the curve start position.
    pub fn start_position(&self) -> Seconds {
        self.start_position
    }

    /// Returns the BPM held before the first segment.
    pub fn start_bpm(&self) -> f64 {
        self.start_bpm
    }

    /// Returns the ordered tempo segments.
    pub fn segments(&self) -> &[TempoCurveSegment] {
        &self.segments
    }

    /// Returns the current tempo in BPM at one transport position in seconds.
    pub fn bpm_at(&self, position: Seconds) -> f64 {
        let position_secs = position.to_seconds_f64();
        let start_secs = self.start_position.to_seconds_f64();
        if position_secs <= start_secs {
            return self.start_bpm;
        }

        let mut segment_start_secs = start_secs;
        let mut segment_start_bpm = self.start_bpm;
        for segment in &self.segments {
            let duration_secs = segment.duration.to_seconds_f64();
            let segment_end_secs = segment_start_secs + duration_secs;
            if position_secs <= segment_end_secs {
                let normalized =
                    ((position_secs - segment_start_secs) / duration_secs).clamp(0.0, 1.0) as f32;
                let curved = apply_curve(normalized, segment.curve) as f64;
                return segment_start_bpm + (segment.target_bpm - segment_start_bpm) * curved;
            }
            segment_start_secs = segment_end_secs;
            segment_start_bpm = segment.target_bpm;
        }

        segment_start_bpm
    }

    fn beats_at_seconds(&self, position_secs: f64) -> f64 {
        if position_secs <= 0.0 {
            return 0.0;
        }

        let start_secs = self.start_position.to_seconds_f64();
        if position_secs <= start_secs {
            return position_secs * self.start_bpm / 60.0;
        }

        let mut accumulated_beats = start_secs * self.start_bpm / 60.0;
        let mut segment_start_secs = start_secs;
        let mut segment_start_bpm = self.start_bpm;
        for segment in &self.segments {
            let duration_secs = segment.duration.to_seconds_f64();
            let segment_end_secs = segment_start_secs + duration_secs;
            if position_secs <= segment_end_secs {
                let elapsed_secs = position_secs - segment_start_secs;
                accumulated_beats +=
                    beats_in_segment(segment_start_bpm, segment, elapsed_secs.max(0.0));
                return accumulated_beats;
            }
            accumulated_beats += beats_in_segment(segment_start_bpm, segment, duration_secs);
            segment_start_secs = segment_end_secs;
            segment_start_bpm = segment.target_bpm;
        }

        accumulated_beats + (position_secs - segment_start_secs) * segment_start_bpm / 60.0
    }

    fn seconds_for_beats(&self, beats: f64) -> f64 {
        if beats <= 0.0 {
            return 0.0;
        }

        let start_secs = self.start_position.to_seconds_f64();
        let start_beats = start_secs * self.start_bpm / 60.0;
        if beats <= start_beats {
            return beats * 60.0 / self.start_bpm;
        }

        let mut accumulated_beats = start_beats;
        let mut segment_start_secs = start_secs;
        let mut segment_start_bpm = self.start_bpm;
        for segment in &self.segments {
            let duration_secs = segment.duration.to_seconds_f64();
            let segment_beats = beats_in_segment(segment_start_bpm, segment, duration_secs);
            if beats <= accumulated_beats + segment_beats {
                let target_beats = beats - accumulated_beats;
                let elapsed_secs =
                    solve_segment_seconds_for_beats(segment_start_bpm, segment, target_beats);
                return segment_start_secs + elapsed_secs;
            }
            accumulated_beats += segment_beats;
            segment_start_secs += duration_secs;
            segment_start_bpm = segment.target_bpm;
        }

        segment_start_secs + (beats - accumulated_beats) * 60.0 / segment_start_bpm
    }
}

/// A map detailing tempo changes such that a [`Beats`] value can be
/// mapped to a deterministic time in seconds (wall clock time). The timestamps
/// are stored as [`Beats`] in absolute time from the start (0 beats), not
/// as relative section lengths.
///
/// Must always have a first [`TempoChange`] at the start, otherwise it wouldn't
/// be possible to map [`Beats`] to seconds. The tempo changes must be
/// sorted in ascending order.
///
/// When a [`TempoCurve`] is present, it becomes the active transport tempo
/// mapping used by the transport and beat conversions.
#[derive(Clone, Debug, PartialEq)]
pub struct MusicalTimeMap {
    tempo_changes: Vec<(TempoChange, Beats)>,
    tempo_curve: Option<TempoCurve>,
}

impl MusicalTimeMap {
    /// Make a new [`MusicalTimeMap`] with a single BPM tempo value of 60 bpm at time 0.
    pub fn new() -> Self {
        Default::default()
    }

    /// Insert a new [`TempoChange`]. If there is a timestamp collision the
    /// existing [`TempoChange`] will be replaced. This will also sort the list
    /// of tempo changes.
    ///
    /// # Example
    /// ```
    /// use knyst::scheduling::{MusicalTimeMap, TempoChange};
    /// use knyst::time::Beats;
    /// let mut map = MusicalTimeMap::new();
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(999, 200000));
    /// map.insert(TempoChange::NewTempo { bpm: 50.0 }, Beats::new(504, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 34.1 }, Beats::new(5, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 642.999 }, Beats::new(5, 201));
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(5, 200));
    /// map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(2000, 0));
    /// map.insert(TempoChange::NewTempo { bpm: 80.0 }, Beats::new(0, 0));
    /// assert!(map.is_sorted());
    /// assert_eq!(map.len(), 6);
    /// ```
    pub fn insert(&mut self, tempo_change: TempoChange, time_stamp: Beats) {
        let mut same_timestamp_index = None;
        for (index, (_change, time)) in self.tempo_changes.iter().enumerate() {
            if *time == time_stamp {
                same_timestamp_index = Some(index);
                break;
            }
        }
        if let Some(index) = same_timestamp_index {
            self.tempo_changes[index] = (tempo_change, time_stamp);
        } else {
            self.tempo_changes.push((tempo_change, time_stamp));
            self.tempo_changes
                .sort_by_key(|(_change, timestamp)| *timestamp);
        }
    }

    /// Remove one [`TempoChange`]. Will insert the default 60 BPM tempo change
    /// at the start if the first tempo change is removed.
    pub fn remove(&mut self, index: usize) {
        self.tempo_changes.remove(index);
        if self.tempo_changes.is_empty() {
            self.tempo_changes
                .push((TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0)));
        }
        if self.tempo_changes[0].1 != Beats::new(0, 0) {
            self.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0));
        }
    }

    /// Replace one [`TempoChange`].
    pub fn replace(&mut self, index: usize, tempo_change: TempoChange) {
        if index < self.tempo_changes.len() {
            self.tempo_changes[index].0 = tempo_change;
        }
    }

    /// Move one [`TempoChange`] to a new position in beats.
    ///
    /// If the first tempo change is moved, a 60 BPM tempo change will be
    /// inserted at the start.
    pub fn move_tempo_change(&mut self, index: usize, time_stamp: Beats) {
        if index < self.tempo_changes.len() {
            self.tempo_changes[index].1 = time_stamp;
            self.tempo_changes
                .sort_by_key(|(_change, timestamp)| *timestamp);
        }
        if index == 0 && time_stamp != Beats::new(0, 0) {
            self.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0));
        }
    }

    /// Sets the continuous tempo curve used by transport/beat conversion.
    pub fn set_tempo_curve(&mut self, tempo_curve: TempoCurve) {
        self.tempo_curve = Some(tempo_curve);
    }

    /// Clears the continuous tempo curve and falls back to discrete tempo changes.
    pub fn clear_tempo_curve(&mut self) {
        self.tempo_curve = None;
    }

    /// Returns the continuous tempo curve, if present.
    pub fn tempo_curve(&self) -> Option<&TempoCurve> {
        self.tempo_curve.as_ref()
    }

    /// Returns the current transport tempo in BPM at one position in seconds.
    pub fn tempo_bpm_at_seconds(&self, position: Seconds) -> f64 {
        if let Some(tempo_curve) = &self.tempo_curve {
            return tempo_curve.bpm_at(position);
        }
        self.discrete_tempo_bpm_at_seconds(position)
    }

    /// Convert a [`Beats`] timestamp to seconds using this map.
    pub fn musical_time_to_secs_f64(&self, ts: Beats) -> f64 {
        if let Some(tempo_curve) = &self.tempo_curve {
            return tempo_curve.seconds_for_beats(ts.as_beats_f64());
        }
        self.discrete_musical_time_to_secs_f64(ts)
    }

    /// Convert a timestamp in seconds to beats using this map.
    pub fn seconds_to_beats(&self, ts: Seconds) -> Beats {
        if let Some(tempo_curve) = &self.tempo_curve {
            return Beats::from_beats_f64(tempo_curve.beats_at_seconds(ts.to_seconds_f64()));
        }
        self.discrete_seconds_to_beats(ts)
    }

    /// Returns the number of discrete tempo changes.
    pub fn len(&self) -> usize {
        self.tempo_changes.len()
    }

    /// Returns true if there are no discrete tempo changes.
    pub fn is_empty(&self) -> bool {
        self.tempo_changes.is_empty()
    }

    /// Returns true if the tempo changes are in order, false if not. For testing purposes.
    pub fn is_sorted(&self) -> bool {
        let mut last_musical_time = Beats::new(0, 0);
        for &(_, musical_time) in &self.tempo_changes {
            if musical_time < last_musical_time {
                return false;
            }
            last_musical_time = musical_time;
        }
        true
    }

    fn discrete_tempo_bpm_at_seconds(&self, position: Seconds) -> f64 {
        let beats = self.discrete_seconds_to_beats(position);
        let mut current_bpm = 60.0;
        for (tempo_change, time) in &self.tempo_changes {
            if *time > beats {
                break;
            }
            let TempoChange::NewTempo { bpm } = tempo_change;
            current_bpm = *bpm;
        }
        current_bpm
    }

    fn discrete_musical_time_to_secs_f64(&self, ts: Beats) -> f64 {
        assert!(!self.tempo_changes.is_empty());
        assert_eq!(self.tempo_changes[0].1, Beats::new(0, 0));
        let mut accumulated_seconds = 0.0;
        let mut duration_remaining = ts;
        for (tempo_change_pair0, tempo_change_pair1) in self
            .tempo_changes
            .iter()
            .zip(self.tempo_changes.iter().skip(1))
        {
            let section_duration = tempo_change_pair1
                .1
                .checked_sub(tempo_change_pair0.1)
                .unwrap();
            if duration_remaining > section_duration {
                accumulated_seconds += tempo_change_pair0.0.to_secs_f64(section_duration);
                duration_remaining = duration_remaining.checked_sub(section_duration).unwrap();
            } else {
                accumulated_seconds += tempo_change_pair0.0.to_secs_f64(duration_remaining);
                duration_remaining = Beats::new(0, 0);
                break;
            }
        }

        if duration_remaining > Beats::new(0, 0) {
            accumulated_seconds += self
                .tempo_changes
                .last()
                .unwrap()
                .0
                .to_secs_f64(duration_remaining);
        }

        accumulated_seconds
    }

    fn discrete_seconds_to_beats(&self, ts: Seconds) -> Beats {
        assert!(!self.tempo_changes.is_empty());
        assert_eq!(self.tempo_changes[0].1, Beats::ZERO);
        let mut accumulated_beats = Beats::ZERO;
        let mut duration_remaining = ts.to_seconds_f64();
        for (tempo_change_pair0, tempo_change_pair1) in self
            .tempo_changes
            .iter()
            .zip(self.tempo_changes.iter().skip(1))
        {
            let section_duration = tempo_change_pair1
                .1
                .checked_sub(tempo_change_pair0.1)
                .unwrap();
            let section_duration_secs = tempo_change_pair0.0.to_secs_f64(section_duration);
            if duration_remaining >= section_duration_secs {
                accumulated_beats += section_duration;
                duration_remaining -= section_duration_secs;
            } else {
                accumulated_beats += tempo_change_pair0.0.secs_f64_to_beats(duration_remaining);
                duration_remaining = 0.0;
                break;
            }
        }

        if duration_remaining > 0.0 {
            accumulated_beats += self
                .tempo_changes
                .last()
                .unwrap()
                .0
                .secs_f64_to_beats(duration_remaining);
        }

        accumulated_beats
    }
}

impl Default for MusicalTimeMap {
    fn default() -> Self {
        Self {
            tempo_changes: vec![(TempoChange::NewTempo { bpm: 60.0 }, Beats::new(0, 0))],
            tempo_curve: None,
        }
    }
}

/// Exposes the shared MusicalTimeMap in a read-only Sync container.
pub struct MusicalTimeMapRef(#[allow(unused)] Arc<RwLock<MusicalTimeMap>>);

fn validate_start(start_position: Seconds, start_bpm: f64) -> Result<(), TempoCurveError> {
    let start_position_seconds = start_position.to_seconds_f64();
    if !start_position_seconds.is_finite() || start_position_seconds < 0.0 {
        return Err(TempoCurveError::Invalid(
            "tempo curve start position must be finite and >= 0".to_string(),
        ));
    }
    if !start_bpm.is_finite() || start_bpm <= 0.0 {
        return Err(TempoCurveError::Invalid(
            "tempo curve start BPM must be finite and > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_segment(duration: Seconds, target_bpm: f64, curve: f32) -> Result<(), TempoCurveError> {
    let duration_seconds = duration.to_seconds_f64();
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err(TempoCurveError::Invalid(
            "tempo curve segment durations must be finite and > 0".to_string(),
        ));
    }
    if !target_bpm.is_finite() || target_bpm <= 0.0 {
        return Err(TempoCurveError::Invalid(
            "tempo curve target BPM values must be finite and > 0".to_string(),
        ));
    }
    if !curve.is_finite() {
        return Err(TempoCurveError::Invalid(
            "tempo curve curves must be finite".to_string(),
        ));
    }
    Ok(())
}

fn beats_in_segment(start_bpm: f64, segment: &TempoCurveSegment, elapsed_secs: f64) -> f64 {
    let duration_secs = segment.duration.to_seconds_f64();
    let clamped_elapsed = elapsed_secs.clamp(0.0, duration_secs);
    let normalized = (clamped_elapsed / duration_secs).clamp(0.0, 1.0) as f32;
    let curved_integral = integrated_curve(normalized, segment.curve) as f64;
    let bpm_delta = segment.target_bpm - start_bpm;
    (start_bpm * clamped_elapsed + bpm_delta * duration_secs * curved_integral) / 60.0
}

fn solve_segment_seconds_for_beats(
    start_bpm: f64,
    segment: &TempoCurveSegment,
    target_beats: f64,
) -> f64 {
    let duration_secs = segment.duration.to_seconds_f64();
    let mut low = 0.0;
    let mut high = duration_secs;
    for _ in 0..48 {
        let mid = (low + high) * 0.5;
        let beats = beats_in_segment(start_bpm, segment, mid);
        if beats < target_beats {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

fn apply_curve(normalized: f32, curve: f32) -> f32 {
    if curve.abs() <= f32::EPSILON {
        return normalized;
    }
    let shaped = 1.0 + curve.abs() * 4.0;
    if curve > 0.0 {
        normalized.powf(shaped)
    } else {
        1.0 - (1.0 - normalized).powf(shaped)
    }
}

fn integrated_curve(normalized: f32, curve: f32) -> f32 {
    if curve.abs() <= f32::EPSILON {
        return 0.5 * normalized * normalized;
    }
    let shaped = 1.0 + curve.abs() * 4.0;
    if curve > 0.0 {
        normalized.powf(shaped + 1.0) / (shaped + 1.0)
    } else {
        normalized - (1.0 - (1.0 - normalized).powf(shaped + 1.0)) / (shaped + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::time::{Beats, Seconds};

    use super::{MusicalTimeMap, TempoChange, TempoCurve, TempoCurveError, TempoCurveSegment};

    #[test]
    fn musical_time_test() {
        let mut map = MusicalTimeMap::new();
        assert_eq!(map.musical_time_to_secs_f64(Beats::new(0, 0)), 0.0);
        assert_eq!(map.musical_time_to_secs_f64(Beats::new(1, 0)), 1.0);
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_fractional_beats::<4>(5, 3)),
            5.75
        );
        assert_eq!(
            map.seconds_to_beats(Seconds::from_seconds_f64(2.0)),
            Beats::from_beats(2)
        );
        map.replace(0, TempoChange::NewTempo { bpm: 120.0 });
        assert_eq!(map.musical_time_to_secs_f64(Beats::new(1, 0)), 0.5);
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_fractional_beats::<4>(5, 3)),
            5.75 * 0.5
        );
        map.insert(TempoChange::NewTempo { bpm: 60.0 }, Beats::from_beats(16));
        map.insert(TempoChange::NewTempo { bpm: 6000.0 }, Beats::from_beats(32));
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(17)),
            16.0 * 0.5 + 1.0
        );
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(32)),
            16.0 * 0.5 + 16.0
        );
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(33)),
            16.0 * 0.5 + 16.0 + 0.01
        );
        assert_eq!(
            map.musical_time_to_secs_f64(Beats::from_beats(32 + 1000)),
            16.0 * 0.5 + 16.0 + 10.0
        );
        assert_eq!(
            map.seconds_to_beats(Seconds::from_seconds_f64(2.0)),
            Beats::from_beats(4)
        );
    }

    #[test]
    fn tempo_curve_rejects_invalid_values() {
        assert!(matches!(
            TempoCurveSegment::new(Seconds::ZERO, 120.0, 0.0),
            Err(TempoCurveError::Invalid(_))
        ));
        assert!(matches!(
            TempoCurveSegment::new(Seconds::from_seconds_f64(1.0), 0.0, 0.0),
            Err(TempoCurveError::Invalid(_))
        ));
        assert!(matches!(
            TempoCurve::new(Seconds::ZERO, 0.0, Vec::new()),
            Err(TempoCurveError::Invalid(_))
        ));
    }

    #[test]
    fn tempo_curve_maps_seconds_and_beats_with_curves() {
        let curve = TempoCurve::new(
            Seconds::ZERO,
            60.0,
            vec![
                TempoCurveSegment::new(Seconds::from_seconds_f64(4.0), 120.0, 0.0)
                    .expect("segment should be valid"),
            ],
        )
        .expect("curve should be valid");
        let mut map = MusicalTimeMap::new();
        map.set_tempo_curve(curve.clone());

        let beats_at_two = map.seconds_to_beats(Seconds::from_seconds_f64(2.0));
        assert!((beats_at_two.as_beats_f64() - 2.5).abs() < 1e-9);

        let seconds_for_two_point_five = map.musical_time_to_secs_f64(Beats::from_beats_f64(2.5));
        assert!((seconds_for_two_point_five - 2.0).abs() < 1e-6);

        assert!((map.tempo_bpm_at_seconds(Seconds::from_seconds_f64(2.0)) - 90.0).abs() < 1e-9);
        assert_eq!(map.tempo_curve(), Some(&curve));
    }

    #[test]
    fn clearing_tempo_curve_falls_back_to_discrete_map() {
        let curve = TempoCurve::new(
            Seconds::ZERO,
            120.0,
            vec![
                TempoCurveSegment::new(Seconds::from_seconds_f64(2.0), 60.0, 0.0)
                    .expect("segment should be valid"),
            ],
        )
        .expect("curve should be valid");
        let mut map = MusicalTimeMap::new();
        map.set_tempo_curve(curve);
        map.clear_tempo_curve();

        assert_eq!(
            map.seconds_to_beats(Seconds::from_seconds_f64(2.0)),
            Beats::from_beats(2)
        );
    }
}
