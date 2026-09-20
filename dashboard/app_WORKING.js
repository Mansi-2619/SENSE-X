/* ============================================================
   SENSE-X ADVANCED COMMAND INTERFACE
   dashboard/app.js

   LIVE BACKEND VERSION

   Expected primary endpoint:
       POST /api/mission/run

   The Python backend should return Observation 1 and, when the
   active-sensing controller requests it, Observation 2.
   ============================================================ */


/* ============================================================
   APPLICATION STATE
   ============================================================ */

let missionRunning = false;
let currentObservation = 0;
let missionData = null;

let uavPosition = {
    x: 0.00,
    y: 0.00,
    z: 10.00
};


/* ============================================================
   GENERAL HELPERS
   ============================================================ */

function get(id) {
    return document.getElementById(id);
}


function setText(id, value) {
    const element = get(id);

    if (element) {
        element.textContent = value;
    }
}


function setWidth(id, percent) {
    const element = get(id);

    if (!element) {
        return;
    }

    const safePercent = Math.max(
        0,
        Math.min(100, Number(percent) || 0)
    );

    element.style.width = `${safePercent}%`;
}


function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


function firstDefined(...values) {
    return values.find(
        value =>
            value !== undefined &&
            value !== null
    );
}


function numberOrZero(value) {
    const n = Number(value);

    return Number.isFinite(n) ? n : 0;
}


/* ============================================================
   BACKEND DATA NORMALIZATION
   ============================================================ */

function normalizeLocalization(raw) {
    if (!raw) {
        return null;
    }

    const x = Number(
        firstDefined(
            raw.x,
            raw.relative_x
        )
    );

    const y = Number(
        firstDefined(
            raw.y,
            raw.relative_y
        )
    );

    if (
        !Number.isFinite(x) ||
        !Number.isFinite(y)
    ) {
        return null;
    }

    return {
        x: x,

        y: y,

        z: numberOrZero(
            firstDefined(
                raw.z,
                raw.relative_z,
                0
            )
        ),

        bearing: numberOrZero(
            firstDefined(
                raw.bearing,
                raw.bearing_deg,
                0
            )
        ),

        range: numberOrZero(
            firstDefined(
                raw.range,
                raw.range_m,
                raw.dataset_radar_range,
                0
            )
        )
    };
}


function normalizeObservation(raw = {}) {
    const thermal =
        typeof raw.thermal === "object"
            ? raw.thermal
            : {};

    const radar =
        typeof raw.radar === "object"
            ? raw.radar
            : {};

    const audio =
        typeof raw.audio === "object"
            ? raw.audio
            : {};

    const fusion =
        raw.fusion ||
        raw.adaptive_fusion ||
        raw.oasf ||
        {};

    const priority =
        raw.priority || {};

    const active =
        raw.active ||
        raw.active_sensing ||
        raw.controller ||
        {};

    const localizationRaw =
        raw.localization ||
        raw.mission_localization ||
        null;


    return {
        thermal: numberOrZero(
            firstDefined(
                thermal.human_probability,
                thermal.evidence,
                thermal.score,
                raw.thermal_evidence,
                typeof raw.thermal === "number"
                    ? raw.thermal
                    : undefined,
                0
            )
        ),

        radar: numberOrZero(
            firstDefined(
                radar.human_probability,
                radar.evidence,
                radar.score,
                raw.radar_evidence,
                raw.radar_human_evidence,
                typeof raw.radar === "number"
                    ? raw.radar
                    : undefined,
                0
            )
        ),

        audio: numberOrZero(
            firstDefined(
                audio.source_present_probability,
                audio.source_probability,
                audio.source_evidence,
                audio.evidence,
                audio.score,
                raw.audio_evidence,
                raw.audio_source_evidence,
                typeof raw.audio === "number"
                    ? raw.audio
                    : undefined,
                0
            )
        ),

        thermalWeight: numberOrZero(
            firstDefined(
                fusion.thermal_weight,
                fusion.weights?.thermal,
                raw.thermal_weight,
                raw.thermalWeight,
                0
            )
        ),

        radarWeight: numberOrZero(
            firstDefined(
                fusion.radar_weight,
                fusion.weights?.radar,
                raw.radar_weight,
                raw.radarWeight,
                0
            )
        ),

        audioWeight: numberOrZero(
            firstDefined(
                fusion.audio_weight,
                fusion.weights?.audio,
                raw.audio_weight,
                raw.audioWeight,
                0
            )
        ),

        fusion: numberOrZero(
            firstDefined(
                fusion.human_score,
                fusion.score,
                fusion.fusion_score,
                raw.fusion_score,
                typeof raw.fusion === "number"
                    ? raw.fusion
                    : undefined,
                0
            )
        ),

        uncertainty: numberOrZero(
            firstDefined(
                fusion.uncertainty,
                raw.uncertainty,
                0
            )
        ),

        fusionDecision: firstDefined(
            fusion.decision,
            raw.fusion_decision,
            raw.fusionDecision,
            raw.decision,
            "WAITING"
        ),

        localizationConfidence: numberOrZero(
            firstDefined(
                localizationRaw?.confidence,
                localizationRaw?.localization_confidence,
                raw.localization_confidence,
                raw.localizationConfidence,
                0
            )
        ),

        priorityScore: numberOrZero(
            firstDefined(
                priority.priority_score,
                priority.score,
                raw.priority_score,
                raw.priorityScore,
                0
            )
        ),

        priorityLevel: firstDefined(
            priority.priority_level,
            priority.level,
            raw.priority_level,
            raw.priorityLevel,
            "STANDBY"
        ),

        action: firstDefined(
            active.action,
            raw.active_sensing_action,
            raw.action,
            "STANDBY"
        ),

        reason: firstDefined(
            active.reason,
            raw.active_sensing_reason,
            raw.reason,
            "No controller reason returned."
        ),

        nextStep: firstDefined(
            active.next_step,
            raw.next_step,
            raw.nextStep,
            "Awaiting controller instruction."
        ),

        explanation: firstDefined(
            fusion.explanation,
            raw.explanation,
            "No fusion explanation returned."
        ),

        localization:
            normalizeLocalization(localizationRaw),

        thermalDetections: numberOrZero(
        firstDefined(
            thermal.detections,
            raw.thermal_detections,
            0
        )
    ),

        thermalImage: firstDefined(
            thermal.image_url,
            thermal.image,
            raw.thermal_image_url,
            null
        ),

        thermalBbox: firstDefined(
            thermal.best_bbox,
            thermal.bbox,
            raw.thermal_bbox,
            null
        ),

        rawThermal: thermal,
        radarPrediction: firstDefined(
            radar.label,
            radar.prediction,
            raw.radar_prediction,
            null
        ),

        audioPrediction: firstDefined(
            audio.label,
            audio.prediction,
            raw.audio_prediction,
            null
        )
    };
}


function normalizeMission(raw = {}) {
    const observation1Raw = firstDefined(
        raw.observation_1,
        raw.observation1,
        raw.first_observation,
        raw.observations?.[0]
    );

    const observation2Raw = firstDefined(
        raw.observation_2,
        raw.observation2,
        raw.second_observation,
        raw.observations?.[1]
    );

    if (!observation1Raw) {
        throw new Error(
            "Backend response does not contain observation_1."
        );
    }

    const finalPosition = firstDefined(
        raw.final_uav_position,
        raw.uav_reposition?.new_position,
        raw.uav_position,
        {
            x: 0,
            y: 2,
            z: 10
        }
    );

    return {
        observation1:
            normalizeObservation(observation1Raw),

        observation2:
            observation2Raw
                ? normalizeObservation(observation2Raw)
                : null,

        movementDistance: numberOrZero(
            firstDefined(
                raw.movement_distance,
                raw.uav_reposition?.movement_distance,
                raw.reposition_distance,
                2.0
            )
        ),

        finalUavPosition: {
            x: numberOrZero(
                firstDefined(
                    finalPosition.x,
                    finalPosition[0],
                    0
                )
            ),

            y: numberOrZero(
                firstDefined(
                    finalPosition.y,
                    finalPosition[1],
                    2
                )
            ),

            z: numberOrZero(
                firstDefined(
                    finalPosition.z,
                    finalPosition[2],
                    10
                )
            )
        }
    };
}


/* ============================================================
   LIVE BACKEND REQUEST
   ============================================================ */

async function fetchMissionData() {
    const endpoints = [
        {
            url: "/api/mission/run",
            method: "POST"
        },

        {
            url: "/api/run-mission",
            method: "POST"
        },

        {
            url: "/api/mission",
            method: "POST"
        }
    ];

    let lastError = null;

    for (const endpoint of endpoints) {
        try {
            const response = await fetch(
                endpoint.url,
                {
                    method: endpoint.method,

                    headers: {
                        "Content-Type":
                            "application/json"
                    }
                }
            );

            if (!response.ok) {
                lastError = new Error(
                    `${endpoint.method} ${endpoint.url} returned HTTP ${response.status}`
                );

                continue;
            }

            const raw = await response.json();

            console.log(
                "SENSE-X backend response:",
                raw
            );

            return normalizeMission(raw);

        } catch (error) {
            lastError = error;
        }
    }

    throw (
        lastError ||
        new Error(
            "Unable to contact SENSE-X mission backend."
        )
    );
}


/* ============================================================
   EVENT LOG
   ============================================================ */

function addLog(
    source,
    message,
    type = "normal"
) {
    const log = get("eventLog");

    if (!log) {
        return;
    }

    const row =
        document.createElement("div");

    row.className =
        `log-entry log-${type}`;

    const now =
        new Date();

    const time =
        now.toLocaleTimeString(
            [],
            {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
            }
        );

    row.innerHTML = `
        <span>${source}</span>
        <time>${time}</time>
        <p>${message}</p>
    `;

    log.appendChild(row);

    log.scrollTop =
        log.scrollHeight;
}


function clearLog() {
    const log = get("eventLog");

    if (!log) {
        return;
    }

    log.innerHTML = "";

    addLog(
        "SYS",
        "Event stream cleared."
    );
}


/* ============================================================
   FLOW CONTROL
   ============================================================ */

function getFlowSteps() {
    return document.querySelectorAll(
        ".flow-step"
    );
}


function resetFlow() {
    getFlowSteps().forEach(step => {
        step.classList.remove(
            "active",
            "complete"
        );
    });
}


function activateFlow(index) {
    const steps = getFlowSteps();

    steps.forEach(
        (step, i) => {
            step.classList.remove(
                "active",
                "complete"
            );

            if (i < index) {
                step.classList.add(
                    "complete"
                );
            }

            if (i === index) {
                step.classList.add(
                    "active"
                );
            }
        }
    );
}


function completeFlow() {
    getFlowSteps().forEach(step => {
        step.classList.remove("active");
        step.classList.add("complete");
    });
}


/* ============================================================
   SENSOR DISPLAY
   ============================================================ */
/* ============================================================
   THERMAL / FLIR DISPLAY
   ============================================================ */

function updateThermalDisplay(observation) {
    const feed = get("thermalFeed");
    const bboxElement = get("thermalBbox");
    const bboxLabel = get("thermalBboxLabel");
    const noDetection = get("thermalNoDetection");

    if (!observation) {
        if (feed) {
            feed.removeAttribute("src");
        }

        if (bboxElement) {
            bboxElement.style.display = "none";
        }

        setText("thermalFeedStatus", "STANDBY");
        setText("thermalFrameName", "NO FRAME");
        setText("thermalDetectionCount", "DET 0");
        setText("thermalConfidenceHud", "CONF 0.0000");

        if (noDetection) {
            noDetection.style.display = "block";
            noDetection.textContent =
                "AWAITING THERMAL OBSERVATION";
        }

        return;
    }

    const thermal = observation.rawThermal || {};

    const imageUrl =
        observation.thermalImage ||
        thermal.image_url ||
        thermal.image ||
        null;

    const detections = numberOrZero(
        firstDefined(
            observation.thermalDetections,
            thermal.detections,
            0
        )
    );

    const confidence = numberOrZero(
        firstDefined(
            observation.thermal,
            thermal.human_probability,
            thermal.evidence,
            thermal.confidence,
            0
        )
    );

    const frameName =
        thermal.image_name ||
        (
            imageUrl
                ? imageUrl.split("/").pop()
                : "NO FRAME"
        );

    setText(
        "thermalFrameName",
        frameName
    );

    setText(
        "thermalDetectionCount",
        `DET ${Math.round(detections)}`
    );

    setText(
        "thermalConfidenceHud",
        `CONF ${confidence.toFixed(4)}`
    );

    setText(
        "thermalFeedStatus",
        detections > 0
            ? "TARGET ACQUIRED"
            : "SCANNING"
    );

    /* --------------------------------------------------------
       THERMAL IMAGE
       -------------------------------------------------------- */

    if (feed && imageUrl) {
        feed.src =
            imageUrl +
            (
                imageUrl.includes("?")
                    ? "&"
                    : "?"
            ) +
            `t=${Date.now()}`;

        feed.onload = () => {
            console.log(
                "SENSE-X thermal frame loaded:",
                imageUrl
            );
        };

        feed.onerror = () => {
            console.error(
                "SENSE-X thermal frame failed:",
                imageUrl
            );

            setText(
                "thermalFeedStatus",
                "FRAME ERROR"
            );
        };
    }

    /* --------------------------------------------------------
       FIND BEST BOUNDING BOX
       -------------------------------------------------------- */

    let bbox = null;
    let bboxConfidence = confidence;

    if (
        Array.isArray(thermal.boxes) &&
        thermal.boxes.length > 0
    ) {
        const best = [...thermal.boxes].sort(
            (a, b) =>
                numberOrZero(b.confidence) -
                numberOrZero(a.confidence)
        )[0];

        if (best) {
            bbox =
                best.bbox ||
                best.box ||
                best.xyxy ||
                null;

            bboxConfidence = numberOrZero(
                firstDefined(
                    best.confidence,
                    confidence
                )
            );
        }
    }

    if (!bbox) {
        bbox =
            thermal.best_bbox ||
            thermal.bbox ||
            observation.thermalBbox ||
            null;
    }

    /* --------------------------------------------------------
       NO DETECTION
       -------------------------------------------------------- */

    if (
        !Array.isArray(bbox) ||
        bbox.length < 4 ||
        detections <= 0
    ) {
        if (bboxElement) {
            bboxElement.style.display = "none";
        }

        if (noDetection) {
            noDetection.style.display = "block";
            noDetection.textContent =
                "NO THERMAL HUMAN DETECTION";
        }

        return;
    }

    if (noDetection) {
        noDetection.style.display = "none";
    }

    /* --------------------------------------------------------
       DRAW YOLO BBOX

       Dataset coordinates:
           x1, y1, x2, y2

       Original FLIR frame:
           640 x 512

       We convert to percentages so resizing the dashboard
       does not break the bounding box.
       -------------------------------------------------------- */

    const imageWidth = numberOrZero(
        firstDefined(
            thermal.image_width,
            640
        )
    ) || 640;

    const imageHeight = numberOrZero(
        firstDefined(
            thermal.image_height,
            512
        )
    ) || 512;

    const x1 = Number(bbox[0]);
    const y1 = Number(bbox[1]);
    const x2 = Number(bbox[2]);
    const y2 = Number(bbox[3]);

    if (
        !Number.isFinite(x1) ||
        !Number.isFinite(y1) ||
        !Number.isFinite(x2) ||
        !Number.isFinite(y2)
    ) {
        if (bboxElement) {
            bboxElement.style.display = "none";
        }

        return;
    }

    const left =
        (x1 / imageWidth) * 100;

    const top =
        (y1 / imageHeight) * 100;

    const width =
        ((x2 - x1) / imageWidth) * 100;

    const height =
        ((y2 - y1) / imageHeight) * 100;

    if (bboxElement) {
        bboxElement.style.display = "block";

        bboxElement.style.left =
            `${left}%`;

        bboxElement.style.top =
            `${top}%`;

        bboxElement.style.width =
            `${width}%`;

        bboxElement.style.height =
            `${height}%`;
    }

    if (bboxLabel) {
        bboxLabel.textContent =
            `HUMAN ${(bboxConfidence * 100).toFixed(1)}%`;
    }
}
function updateSensor(name, value) {
    const numericValue =
        numberOrZero(value);

    const idMap = {
        thermal: {
            value: "thermalEvidence",
            bar: "thermalEvidenceBar",
            status: "thermalStatus"
        },

        radar: {
            value: "radarEvidence",
            bar: "radarEvidenceBar",
            status: "radarStatus"
        },

        audio: {
            value: "audioEvidence",
            bar: "audioEvidenceBar",
            status: "audioStatus"
        }
    };

    const ids = idMap[name];

    if (!ids) {
        console.warn(
            "Unknown SENSE-X sensor:",
            name
        );

        return;
    }

    setText(
        ids.value,
        numericValue.toFixed(4)
    );

    setWidth(
        ids.bar,
        numericValue * 100
    );

    let status = "LOW";

    if (numericValue >= 0.75) {
        status = "STRONG";

    } else if (numericValue >= 0.50) {
        status = "DETECTED";

    } else if (numericValue >= 0.25) {
        status = "WEAK";
    }

    setText(
        ids.status,
        status
    );
}

function resetSensors() {
    updateSensor("thermal", 0);
    updateSensor("radar", 0);
    updateSensor("audio", 0);
}


/* ============================================================
   FUSION DISPLAY
   ============================================================ */

function updateFusion(data) {
    const score =
        numberOrZero(data.fusion);

    setText(
        "fusionScore",
        score.toFixed(4)
    );

    setText(
        "oasfScore",
        score.toFixed(4)
    );

    setText(
        "uncertaintyValue",
        numberOrZero(
            data.uncertainty
        ).toFixed(4)
    );

    setText(
        "oasfDecision",
        data.fusionDecision
    );

    setText(
        "fusionThermal",
        data.thermal.toFixed(4)
    );

    setText(
        "fusionRadar",
        data.radar.toFixed(4)
    );

    setText(
        "fusionAudio",
        data.audio.toFixed(4)
    );

    setText(
        "thermalWeight",
        data.thermalWeight.toFixed(2)
    );

    setText(
        "radarWeight",
        data.radarWeight.toFixed(2)
    );

    setText(
        "audioWeight",
        data.audioWeight.toFixed(2)
    );

    setWidth(
        "thermalWeightBar",
        data.thermalWeight * 100
    );

    setWidth(
        "radarWeightBar",
        data.radarWeight * 100
    );

    setWidth(
        "audioWeightBar",
        data.audioWeight * 100
    );

    setText(
        "fusionExplanation",
        data.explanation
    );

    setText(
        "aiExplanation",
        data.explanation
    );

    setWidth(
        "confidenceFill",
        score * 100
    );

    setText(
        "confidenceValue",
        score.toFixed(4)
    );
}


/* ============================================================
   DECISION BADGE
   ============================================================ */

function updateDecisionBadge(decision) {
    const badge =
        get("decisionBadge");

    if (!badge) {
        return;
    }

    badge.textContent =
        decision;

    badge.classList.remove(
        "confirmed",
        "suspected",
        "standby"
    );

    if (
        decision ===
        "HUMAN_CONFIRMED"
    ) {
        badge.classList.add(
            "confirmed"
        );

    } else if (
        decision ===
        "HUMAN_SUSPECTED"
    ) {
        badge.classList.add(
            "suspected"
        );

    } else {
        badge.classList.add(
            "standby"
        );
    }
}


/* ============================================================
   PRIORITY
   ============================================================ */

function updatePriority(data) {
    const display =
        `${data.priorityLevel} / ` +
        `${data.priorityScore.toFixed(4)}`;

    setText(
        "priorityValue",
        display
    );

    setText(
        "priorityScore",
        data.priorityScore.toFixed(4)
    );

    setText(
        "priorityLevel",
        data.priorityLevel
    );
}


/* ============================================================
   LOCALIZATION
   ============================================================ */

function clearLocalization() {
    setText("bearingValue", "--");
    setText("targetBearing", "ACQUIRING");
    setText("rangeValue", "--");

    setText("coordinateX", "--");
    setText("coordinateY", "--");
    setText("coordinateZ", "--");

    setText(
        "localizationConfidence",
        "0.0000"
    );

    setText(
        "lockStatus",
        "NO LOCK"
    );

    const target =
        get("targetMarker");

    if (target) {
        target.classList.add(
            "hidden-target"
        );
    }

    const line =
        get("bearingLine");

    if (line) {
        line.style.opacity = "0";
    }
}


function updateLocalization(
    localization,
    confidence
) {
    if (!localization) {
        clearLocalization();
        return;
    }

    const bearing =
        `${localization.bearing.toFixed(2)}°`;

    setText(
        "bearingValue",
        bearing
    );

    setText(
        "targetBearing",
        bearing
    );

    setText(
        "rangeValue",
        `${localization.range.toFixed(2)} m`
    );

    setText(
        "coordinateX",
        `${localization.x.toFixed(2)} m`
    );

    setText(
        "coordinateY",
        `${localization.y.toFixed(2)} m`
    );

    setText(
        "coordinateZ",
        `${localization.z.toFixed(2)} m`
    );

    setText(
        "localizationConfidence",
        numberOrZero(
            confidence
        ).toFixed(4)
    );

    setText(
        "lockStatus",
        "TARGET LOCK"
    );

    const target =
        get("targetMarker");

    if (target) {
        target.classList.remove(
            "hidden-target"
        );
    }

    drawBearing(
        localization.bearing,
        localization.range
    );
}


/* ============================================================
   TARGET BEARING VISUALIZATION
   ============================================================ */

function drawBearing(
    bearingDegrees,
    rangeMeters
) {
    const line =
        get("bearingLine");

    const target =
        get("targetMarker");

    if (!line || !target) {
        return;
    }

    const visualAngle =
        bearingDegrees - 90;

    const radians =
        visualAngle *
        Math.PI /
        180;

    const distance =
        Math.min(
            180,
            70 + rangeMeters * 42
        );

    const x =
        Math.cos(radians) *
        distance;

    const y =
        Math.sin(radians) *
        distance;

    target.style.left =
        `calc(50% + ${x}px)`;

    target.style.top =
        `calc(50% + ${y}px)`;

    line.style.width =
        `${distance}px`;

    line.style.transform =
        `rotate(${visualAngle}deg)`;

    line.style.opacity =
        "1";
}


/* ============================================================
   UAV DISPLAY
   ============================================================ */

function updateUAVReadout() {
    const text =
        `X ${uavPosition.x.toFixed(2)} | ` +
        `Y ${uavPosition.y.toFixed(2)} | ` +
        `Z ${uavPosition.z.toFixed(2)}`;

    setText(
        "uavPosition",
        text
    );

    setText(
        "uavCoordinates",
        text
    );

    setText(
        "uavAltitude",
        `${uavPosition.z.toFixed(2)} m`
    );
}


async function repositionUAV() {
    addLog(
        "UAV",
        "Lateral reposition command issued.",
        "warning"
    );

    setText(
        "missionAction",
        "REPOSITIONING"
    );

    setText(
        "systemStatus",
        "UAV REPOSITIONING"
    );

    const marker =
        get("uavMarker");

    const newPosition =
        missionData.finalUavPosition;

    if (marker) {
        marker.style.left = "58%";
    }

    await sleep(900);

    uavPosition = {
        x: newPosition.x,
        y: newPosition.y,
        z: newPosition.z
    };

    updateUAVReadout();

    addLog(
        "UAV",
        `Reposition complete. Movement distance ` +
        `${missionData.movementDistance.toFixed(2)} m.`,
        "good"
    );

    setText(
        "systemStatus",
        "RESCAN ACQUISITION"
    );

    await sleep(500);
}


/* ============================================================
   ACTIVE SENSING DISPLAY
   ============================================================ */

function updateAction(data) {
    const action =
        get("missionAction");

    if (action) {
        action.textContent =
            data.action;

        action.classList.remove(
            "action-rescan",
            "action-investigate"
        );

        if (
            data.action ===
            "REPOSITION_AND_RESCAN"
        ) {
            action.classList.add(
                "action-rescan"
            );

        } else if (
            data.action ===
            "INVESTIGATE"
        ) {
            action.classList.add(
                "action-investigate"
            );
        }
    }

    setText(
        "missionReason",
        data.reason
    );

    setText(
        "nextStep",
        data.nextStep
    );
}


/* ============================================================
   OBSERVATION DISPLAY
   ============================================================ */

async function displayObservation(
    observationNumber,
    data
) {
    currentObservation =
        observationNumber;

    setText(
        "observationNumber",
        `OBSERVATION 0${observationNumber}`
    );

    setText(
        "systemStatus",
        `OBSERVATION 0${observationNumber}`
    );

    resetFlow();

    activateFlow(0);

    addLog(
        "SYS",
        `Observation ${observationNumber} initiated.`
    );

    await sleep(350);


    /* ---------------- THERMAL ---------------- */

    addLog(
        "THM",
        "Thermal inference result received."
    );

    updateSensor(
        "thermal",
        data.thermal
    );

    if (data.thermal > 0) {
        addLog(
            "THM",
            `Human evidence ${data.thermal.toFixed(4)}.`,
            "good"
        );

    } else {
        addLog(
            "THM",
            "No thermal human detection.",
            "warning"
        );
    }

    await sleep(450);


    /* ---------------- RADAR ---------------- */

    activateFlow(1);

    addLog(
        "RDR",
        "Radar inference result received."
    );

    updateSensor(
        "radar",
        data.radar
    );

    addLog(
        "RDR",
        `Human evidence ${data.radar.toFixed(4)}.`,
        data.radar >= 0.5
            ? "good"
            : "warning"
    );

    await sleep(450);


    /* ---------------- AUDIO ---------------- */

    activateFlow(2);

    addLog(
        "AUD",
        "Acoustic inference result received."
    );

    updateSensor(
        "audio",
        data.audio
    );

    addLog(
        "AUD",
        `Source evidence ${data.audio.toFixed(4)}.`
    );

    await sleep(450);


    /* ---------------- FUSION ---------------- */

    activateFlow(3);

    setText(
        "systemStatus",
        "OASF FUSION"
    );

    addLog(
        "OASF",
        "Displaying backend fusion result."
    );

    await sleep(400);

    updateFusion(data);

    updateDecisionBadge(
        data.fusionDecision
    );

    updatePriority(data);

    addLog(
        "OASF",
        `Fusion ${data.fusion.toFixed(4)} / ` +
        `${data.fusionDecision}.`,
        data.fusionDecision ===
            "HUMAN_CONFIRMED"
            ? "good"
            : "warning"
    );

    await sleep(400);


    /* ---------------- LOCALIZATION ---------------- */

    activateFlow(4);

    if (data.localization) {
        setText(
            "systemStatus",
            "TARGET LOCALIZATION"
        );

        updateLocalization(
            data.localization,
            data.localizationConfidence
        );

        addLog(
            "LOC",
            `Target range ` +
            `${data.localization.range.toFixed(2)} m, ` +
            `bearing ` +
            `${data.localization.bearing.toFixed(2)}°.`,
            "good"
        );

    } else {
        clearLocalization();

        addLog(
            "LOC",
            "Target localization unavailable.",
            "warning"
        );
    }

    await sleep(400);


    /* ---------------- ACTIVE SENSING ---------------- */

    activateFlow(5);

    updateAction(data);

    addLog(
        "ACT",
        `Controller action: ${data.action}.`,
        data.action === "INVESTIGATE"
            ? "good"
            : "warning"
    );

    await sleep(450);
}


/* ============================================================
   MISSION EXECUTION
   ============================================================ */

async function startMission() {
    if (missionRunning) {
        return;
    }

    missionRunning = true;

    const button =
        get("startMissionButton");

    if (button) {
        button.disabled = true;
        button.textContent =
            "RUNNING BACKEND";
    }

    resetDashboard();

    setText(
        "systemStatus",
        "BACKEND INFERENCE"
    );

    addLog(
        "SYS",
        "SENSE-X autonomous mission requested.",
        "good"
    );

    addLog(
        "SYS",
        "Waiting for thermal, radar, audio and OASF pipeline."
    );

    try {
        missionData =
            await fetchMissionData();

        console.log(
            "Normalized mission:",
            missionData
        );

        setText(
            "systemStatus",
            "MISSION ACTIVE"
        );

        addLog(
            "SYS",
            "Real backend mission result received.",
            "good"
        );

        await sleep(500);

        /* ==========================================
           OBSERVATION 1
           ========================================== */

        await displayObservation(
            1,
            missionData.observation1
        );

        const firstAction =
            missionData.observation1.action;

        /* ==========================================
           CLOSED LOOP
           ========================================== */

        if (
            firstAction ===
            "REPOSITION_AND_RESCAN"
        ) {
            if (!missionData.observation2) {
                throw new Error(
                    "Controller requested REPOSITION_AND_RESCAN but backend returned no observation_2."
                );
            }

            addLog(
                "SYS",
                "Controller requested another observation.",
                "warning"
            );

            await repositionUAV();

            await displayObservation(
                2,
                missionData.observation2
            );

        } else {
            addLog(
                "SYS",
                "Controller did not request a rescan.",
                "good"
            );
        }


        /* ==========================================
           FINAL RESULT
           ========================================== */

        const finalObservation =
            missionData.observation2 ||
            missionData.observation1;

        updateSensor(
            "thermal",
            finalObservation.thermal
        );

        updateSensor(
            "radar",
            finalObservation.radar
        );

        updateSensor(
            "audio",
            finalObservation.audio
        );
        updateThermalDisplay(
            finalObservation
        );
        updateFusion(
            finalObservation
        );

        updatePriority(
            finalObservation
        );

        updateLocalization(
            finalObservation.localization,
            finalObservation.localizationConfidence
        );

        updateAction(
            finalObservation
        );
        
        updateDecisionBadge(
            finalObservation.fusionDecision
        );

        completeFlow();

        setText(
            "systemStatus",
            finalObservation.fusionDecision ===
                "HUMAN_CONFIRMED"
                ? "TARGET CONFIRMED"
                : finalObservation.fusionDecision
        );

        addLog(
            "SYS",
            "Closed-loop sensing sequence complete.",
            "good"
        );

        addLog(
            "SYS",
            `Final decision: ` +
            `${finalObservation.fusionDecision}.`,
            finalObservation.fusionDecision ===
                "HUMAN_CONFIRMED"
                ? "good"
                : "warning"
        );

        addLog(
            "SYS",
            `Final action: ` +
            `${finalObservation.action}.`,
            "good"
        );

        if (button) {
            button.textContent =
                "REPLAY MISSION";
        }

    } catch (error) {
        console.error(
            "SENSE-X mission error:",
            error
        );

        setText(
            "systemStatus",
            "BACKEND ERROR"
        );

        setText(
            "missionAction",
            "CONNECTION ERROR"
        );

        setText(
            "missionReason",
            "The dashboard could not obtain a valid mission result from the Python backend."
        );

        setText(
            "nextStep",
            "Check /api/mission/run and the FastAPI terminal."
        );

        addLog(
            "ERR",
            error.message ||
            String(error),
            "warning"
        );

        if (button) {
            button.textContent =
                "RETRY MISSION";
        }

    } finally {
        if (button) {
            button.disabled = false;
        }

        missionRunning = false;
    }
}


/* ============================================================
   RESET DASHBOARD
   ============================================================ */

function resetDashboard() {
    currentObservation = 0;
    missionData = null;

    uavPosition = {
        x: 0.00,
        y: 0.00,
        z: 10.00
    };

    resetSensors();
    resetFlow();
    clearLocalization();

    setText(
        "fusionScore",
        "0.0000"
    );

    setText(
        "oasfScore",
        "0.0000"
    );

    setText(
        "uncertaintyValue",
        "0.0000"
    );

    setText(
        "confidenceValue",
        "0.0000"
    );

    setWidth(
        "confidenceFill",
        0
    );

    setText(
        "oasfDecision",
        "WAITING"
    );

    updateDecisionBadge(
        "WAITING"
    );

    setText(
        "priorityValue",
        "STANDBY"
    );

    setText(
        "priorityScore",
        "0.0000"
    );

    setText(
        "priorityLevel",
        "STANDBY"
    );

    setText(
        "thermalWeight",
        "0.00"
    );

    setText(
        "radarWeight",
        "0.00"
    );

    setText(
        "audioWeight",
        "0.00"
    );

    setWidth(
        "thermalWeightBar",
        0
    );

    setWidth(
        "radarWeightBar",
        0
    );

    setWidth(
        "audioWeightBar",
        0
    );

    setText(
        "fusionThermal",
        "0.0000"
    );

    setText(
        "fusionRadar",
        "0.0000"
    );

    setText(
        "fusionAudio",
        "0.0000"
    );

    setText(
        "missionAction",
        "STANDBY"
    );

    setText(
        "missionReason",
        "Awaiting mission execution."
    );

    setText(
        "nextStep",
        "Initialize autonomous sensing."
    );

    setText(
        "observationNumber",
        "STANDBY"
    );

    setText(
        "fusionExplanation",
        "Awaiting multimodal sensor evidence."
    );

    setText(
        "aiExplanation",
        "Awaiting multimodal sensor evidence."
    );

    setText(
        "systemStatus",
        "SYSTEM READY"
    );

    updateUAVReadout();

    const marker =
        get("uavMarker");

    if (marker) {
        marker.style.left = "50%";
        marker.style.top = "50%";
    }

    const log =
        get("eventLog");

    if (log) {
        log.innerHTML = "";
    }
}


/* ============================================================
   BACKEND HEALTH CHECK
   ============================================================ */

async function checkBackendHealth() {
    try {
        const response =
            await fetch(
                "/api/health",
                {
                    cache: "no-store"
                }
            );

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const result =
            await response.json();

        console.log(
            "SENSE-X backend:",
            result
        );

        addLog(
            "NET",
            "Python backend online.",
            "good"
        );

    } catch (error) {
        console.warn(
            "Backend health check failed:",
            error
        );

        addLog(
            "NET",
            "Python backend health check failed.",
            "warning"
        );
    }
}


/* ============================================================
   CLOCK
   ============================================================ */

function updateClock() {
    const now =
        new Date();

    setText(
        "missionClock",
        now.toLocaleTimeString(
            [],
            {
                hour12: false
            }
        )
    );
}


/* ============================================================
   SYSTEM INITIALIZATION
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {
        resetDashboard();

        updateClock();

        setInterval(
            updateClock,
            1000
        );

        const startButton =
            get("startMissionButton");

        if (startButton) {
            startButton.addEventListener(
                "click",
                startMission
            );

        } else {
            console.error(
                "SENSE-X: startMissionButton was not found."
            );
        }

        const clearButton =
            get("clearLogButton");

        if (clearButton) {
            clearButton.addEventListener(
                "click",
                clearLog
            );
        }

        addLog(
            "SYS",
            "SENSE-X command interface initialized."
        );

        addLog(
            "SYS",
            "Thermal perception module standing by."
        );

        addLog(
            "SYS",
            "Radar perception module standing by."
        );

        addLog(
            "SYS",
            "Acoustic perception module standing by."
        );

        addLog(
            "SYS",
            "OASF fusion engine standing by."
        );

        addLog(
            "SYS",
            "Autonomous sensing controller ready.",
            "good"
        );

        checkBackendHealth();

        console.log(
            "SENSE-X live dashboard initialized."
        );
    }
);