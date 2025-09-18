# AI Weekend Travel Buddy

🎒 **Cypher 2025 Hackathon Project**

An AI-powered **weekend travel buddy** that generates a **personalized, budget-friendly, and engaging 2-day itinerary** for any destination.  

---

## ✨ Features
- **Multi-Agent System**
  - **Explorer Agent** → Finds & ranks attractions (opening hours, entry fees, travel times).  
  - **Food Agent** → Suggests local restaurants, cafes & street food near attractions.  
  - **Budget Agent** → Splits budget across stay, food, activities & travel; keeps plan realistic.  
- **2-Day Optimized Itinerary** with morning/afternoon/evening slots.  
- **Budget Awareness** → Estimates costs to ensure spending stays within budget.  
- **Online + Offline Support**  
  - Online: Live POIs from OpenStreetMap + travel times via OSRM.  
  - Offline: Demo mode with sample data (no internet needed).  
- **Outputs** → Human-readable plan (console) + JSON file (`itinerary.json`).  

---

## ⚡ Installation
```bash
git clone <your-repo>
cd ai-weekend-travel-buddy
pip install requests
```

---

## 🚀 Usage
```bash
python main.py
```

---

## 📁 Project Structure
```
ai-weekend-travel-buddy/
├── main.py                 # Entry point
├── agents/                 # Multi-agent system
│   ├── explorer_agent.py   # Finds attractions
│   ├── food_agent.py       # Suggests restaurants  
│   └── budget_agent.py     # Manages budget
├── utils/                  # Helper functions
│   ├── osm_api.py         # OpenStreetMap integration
│   └── osrm_api.py        # Routing & travel times
└── itinerary.json         # Generated output
```

---

## 🏆 Hackathon Highlights
- **Multi-Agent Architecture** for specialized travel planning
- **Real-time Data Integration** with OpenStreetMap & OSRM
- **Budget-Conscious Planning** ensuring affordable experiences
- **Offline Capability** for areas with limited connectivity