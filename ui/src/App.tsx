import { BrowserRouter, Routes, Route, Navigate } from "react-router";
import LoginPage from "@/pages/LoginPage";
import ConferencePage from "@/pages/ConferencePage";
import AppLayout from "@/layouts/AppLayout";
import TeamPage from "@/pages/TeamPage";
import BattingTab from "@/pages/BattingTab";
import PitchingTab from "@/pages/PitchingTab";
import PlayerPage from "@/pages/PlayerPage";
import PercentilesTab from "@/pages/player/PercentilesTab";
import StatsTab from "@/pages/player/StatsTab";
import HeatMapTab from "@/pages/player/HeatMapTab";
import RequireAuth from "@/utils/supabase/requireauth";
import PublicOnly from "@/utils/supabase/publiconly";
import ResetPasswordPage from "@/pages/ResetPasswordPage";
import TeamPerformancePage from "@/pages/TeamPerformancePage";

const basename = import.meta.env.BASE_URL.replace(/\/$/, "");

export default function App() {
  return (
    <BrowserRouter basename={basename}>
      <Routes>
        {/* Public-only group: if signed in, redirect to /conferences */}
        <Route element={<PublicOnly />}>
          <Route index element={<LoginPage />} />
          <Route path="reset-password" element={<ResetPasswordPage />} />
        </Route>

        {/* Auth-only group */}
        <Route element={<RequireAuth />}>
          <Route element={<AppLayout />}>
            <Route path="conferences" element={<ConferencePage />} />
            
            {/* Player-level pages */}
            <Route
              path="team/:trackmanAbbreviation/player/:playerName"
              element={<PlayerPage />}
            >
              <Route path="stats/:year" element={<StatsTab />} />
              <Route path="heat-map/:year" element={<HeatMapTab />} />
              <Route path="percentiles/:year" element={<PercentilesTab />} />
            </Route>

            {/* Team-level pages */}
            <Route path="team/:trackmanAbbreviation" element={<TeamPage />}>
              {/* Default redirect now goes to Batting instead of Roster */}
              <Route index element={<Navigate to="batting" replace />} />
              <Route path="batting" element={<BattingTab />} />
              <Route path="pitching" element={<PitchingTab />} />
            </Route>

            {/* Team performance overview */}
            <Route path="teamperformance" element={<TeamPerformancePage />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
