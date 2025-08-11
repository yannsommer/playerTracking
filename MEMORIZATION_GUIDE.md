# Football Analytics Codebase - Memorization Guide

## 🏗️ **Project Structure (Remember This Layout)**

```
dfl/
├── LaurieOnTracking/
│   ├── descriptive_analysis.py    # Statistical analysis & visualizations
│   ├── lstm.py                    # Basic ML models (LSTM/Transformer)  
│   └── advanced_pass_prediction.py # Sophisticated multi-modal architecture
└── sample-data/
    ├── data/ (3 games)           # CSV tracking/event data
    └── documentation/            # Event definitions (PDF)
```

## 🎯 **Core Concepts to Memorize**

### Data Format (Metrica Sports)
- **Coordinates**: (0,0) = top-left, (1,1) = bottom-right, (0.5,0.5) = center
- **Field**: 105x68 meters, normalized to 0-1 scale
- **Files**: RawEventsData.csv + RawTrackingData_Home/Away_Team.csv

### Three Analysis Approaches
1. **descriptive_analysis.py** → Statistical analysis (networks, heatmaps, zones)
2. **lstm.py** → Basic ML (simple LSTM/Transformer, 8 features, window=8)
3. **advanced_pass_prediction.py** → Advanced ML (multi-modal, 13 features, dual windows)

## 🧠 **Advanced Model Architecture - Key Memory Points**

### Multi-Modal Components (Remember the 5 pillars)
1. **Enhanced Event Encoder** - Categorical + Numerical + Spatial + Tactical fusion
2. **Multi-Scale Positional Encoding** - 3 scales [1x, 4x, 16x] for different temporal patterns
3. **Spatial Relationship Encoder** - Distance/angle between players and positions
4. **Tactical Context Encoder** - Game phase, field zones, pressure indicators
5. **Player Relationship GNN** - Graph neural network for player interactions

### Attention Mechanisms
- **AdaptiveMultiHeadAttention** - Dynamic head importance weighting
- **Multi-scale temporal pooling** - Global avg + Global max + Local patterns
- **Cross-modal attention** - Between event sequences and tactical context

### Feature Engineering (13 vs 8 basic)
**Basic (8)**: start_x, start_y, end_x, end_y, pass_len, pass_ang, event_dt, period
**Advanced (+5)**: minute, time_pressure, pass_difficulty, forward_pass, progressive_pass
**Tactical**: game_phase, zone_x, zone_y (categorical embeddings)

## 📋 **Command Templates to Memorize**

### Descriptive Analysis
```bash
python LaurieOnTracking/descriptive_analysis.py \
    --csv sample-data/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv \
    --team All --grid_x 6 --grid_y 4 --top_edges 25
```

### Basic ML Model
```bash
python LaurieOnTracking/lstm.py \
    --csv [path] --model lstm --epochs 8 --window 8
```

### Advanced ML Model
```bash
python LaurieOnTracking/advanced_pass_prediction.py \
    --csv [path] --model advanced_transformer --epochs 20 \
    --window 12 --context_window 24 --batch_size 64 --lr 1e-4
```

## 🔥 **Key Technical Differentiators**

### Basic Model Limitations
- Simple embedding + LSTM/Transformer
- Single 8-event window
- 8 basic features only
- No tactical awareness
- No player relationship modeling

### Advanced Model Advantages  
- **6-layer transformer** with 256-dim embeddings
- **Dual windows**: 12 recent + 24 context events
- **13 enhanced features** with tactical context
- **GNN player modeling** with adjacency matrices
- **Multi-scale attention** with adaptive importance
- **Spatial relationship encoding** for positioning
- **Label smoothing + AdamW + Cosine scheduler**

## 🎖️ **Performance Expectations**

### Metrics Tracked
- **Top-1 Accuracy**: Exact pass recipient prediction
- **Top-3 Accuracy**: Recipient in top 3 predictions  
- **Top-5 Accuracy**: Recipient in top 5 predictions

### Expected Improvements (Advanced vs Basic)
- **Accuracy**: ~15-25% improvement on Top-1
- **Top-3/Top-5**: ~20-30% improvement
- **Tactical Understanding**: Significantly better context awareness
- **Player Relationships**: Better modeling of passing networks

## 💡 **Architecture Memory Tricks**

### Remember "STAGE" for Advanced Model Flow
- **S**patial encoding (distance/angle relationships)  
- **T**actical context (game phase, zones, pressure)
- **A**ttention mechanisms (adaptive multi-head)
- **G**NN player modeling (relationship graphs)
- **E**vent encoding (enhanced multi-modal fusion)

### Remember "3-2-1" Pattern
- **3 temporal scales** in positional encoding
- **2 window sizes** (recent=12, context=24)  
- **1 unified prediction** head with multi-modal fusion

### Remember Key Numbers
- **256** embedding dimensions (vs 64 basic)
- **6** transformer layers (vs 1-2 basic)
- **8** attention heads with adaptive weighting
- **13** enhanced features (vs 8 basic)
- **24** context window (vs 8 basic)

## 🚀 **When to Use Which Model**

### Use descriptive_analysis.py when:
- Need statistical insights and visualizations
- Want pass networks, heatmaps, zone analysis
- Exploring data patterns and team tactics

### Use lstm.py when:
- Quick prototyping and baseline results
- Limited computational resources
- Simple pass prediction tasks

### Use advanced_pass_prediction.py when:
- Need state-of-the-art prediction accuracy
- Have sufficient computational resources (GPU recommended)
- Want tactical and spatial awareness in predictions
- Building production football analytics systems