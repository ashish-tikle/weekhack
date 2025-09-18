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
# weekhack
