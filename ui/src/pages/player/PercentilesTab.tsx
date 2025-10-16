import React, { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import StatBar from "@/components/player/StatBar";
import { supabase } from "@/utils/supabase/client";

const boxStyle: React.CSSProperties = {
  flex: 1,
  minHeight: "400px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  margin: "0 12px",
};

// Infield Spray Chart component
const InfieldSprayChart: React.FC<{ stats: any }> = ({ stats }) => {
  if (!stats) return null;

  const counts = [
    stats.infield_left_slice ?? 0,
    stats.infield_lc_slice ?? 0,
    stats.infield_center_slice ?? 0,
    stats.infield_rc_slice ?? 0,
    stats.infield_right_slice ?? 0,
  ];

  const total = counts.reduce((a, b) => a + b, 0);
  const percents = total > 0 ? counts.map((c) => (c / total) * 100) : counts;
  const maxPercent = Math.max(...percents, 0);

  const getColor = (percent: number) => {
    if (maxPercent === 0) return "rgb(255,255,255)";
    const intensity = percent / 50;
    const red = 255;
    const green = Math.round(255 * (1 - intensity));
    const blue = Math.round(255 * (1 - intensity));
    return `rgb(${red},${green},${blue})`;
  };

  const slices = [
    { label: "Left", percent: percents[0] ?? 0 },
    { label: "Left-Center", percent: percents[1] ?? 0 },
    { label: "Center", percent: percents[2] ?? 0 },
    { label: "Right-Center", percent: percents[3] ?? 0 },
    { label: "Right", percent: percents[4] ?? 0 },
  ];

  return (
    <svg viewBox="-150 -50 300 200" style={{ width: "100%", maxWidth: 400, margin: "auto", transform: "scale(1, -1)" }}>
      {slices.map((slice, i) => {
        const startAngle = -45 + i * 18;
        const endAngle = startAngle + 18;
        const radius = 120;

        const x1 = radius * Math.sin((Math.PI / 180) * startAngle);
        const y1 = radius * Math.cos((Math.PI / 180) * startAngle);
        const x2 = radius * Math.sin((Math.PI / 180) * endAngle);
        const y2 = radius * Math.cos((Math.PI / 180) * endAngle);

        const d = `M 0 0 L ${x1} ${y1} A ${radius} ${radius} 0 0 1 ${x2} ${y2} Z`;
        const color = getColor(slice.percent);
        const label = `${slice.percent.toFixed(0)}%`;
        const midAngle = (startAngle + endAngle) / 2;
        const labelX = (radius + 15) * Math.sin((Math.PI / 180) * midAngle);
        const labelY = (radius + 15) * Math.cos((Math.PI / 180) * midAngle);

        return (
          <g key={slice.label}>
            <path d={d} fill={color} stroke="#333" strokeWidth={1} />
            <text transform="scale(1, -1)" x={labelX} y={-labelY} textAnchor="middle" alignmentBaseline="middle" fontSize={12} fill="#222" fontWeight="bold">
              {label}
            </text>
          </g>
        );
      })}
      <circle cx={0} cy={0} r={4} fill="#333" transform="scale(1, -1)" />
    </svg>
  );
};

export default function PercentilesTab() {
  const { trackmanAbbreviation, playerName, year } = useParams<{
    trackmanAbbreviation: string;
    playerName: string;
    year: string;
  }>();
  const [searchParams] = useSearchParams();

  const roleParam = searchParams.get("role");
  const role: "batter" | "pitcher" | null =
    roleParam === "batter" || roleParam === "pitcher" ? roleParam : null;

  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchStats() {
      if (!role) {
        setError("Please specify a valid role in the URL query parameter: ?role=batter or ?role=pitcher");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const safeYear = year || "2025";
        const formattedPlayerName = playerName ? decodeURIComponent(playerName).replace("_", ", ") : "";
        const decodedTeamName = trackmanAbbreviation ? decodeURIComponent(trackmanAbbreviation) : "";

        if (role === "batter") {
          const { data, error } = await supabase
            .from("AdvancedBattingStats")
            .select("*")
            .eq("BatterTeam", decodedTeamName)
            .eq("Year", safeYear);

          if (error) throw error;

          const playerStats = data.find((b: any) => b.Batter === formattedPlayerName);
          setStats(playerStats);
        } else if (role === "pitcher") {
          const { data, error } = await supabase
            .from("AdvancedPitchingStats")
            .select("*")
            .eq("PitcherTeam", decodedTeamName)
            .eq("Year", safeYear);

          if (error) throw error;

          const playerStats = data.find((p: any) => p.Pitcher === formattedPlayerName);
          setStats(playerStats);
        }
      } catch (err: any) {
        console.error(err);
        setError("Failed to load player stats");
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
  }, [trackmanAbbreviation, playerName, year, role]);

  const getRankColor = (rank: number): string => {
    const r = Math.max(0, Math.min(rank, 100));
    const blueRGB = { r: 0, g: 123, b: 255 };
    const greyRGB = { r: 153, g: 153, b: 153 };
    const t = Math.abs(r - 50) / 50;
    const rVal = Math.round(greyRGB.r + t * (blueRGB.r - greyRGB.r));
    const gVal = Math.round(greyRGB.g + t * (blueRGB.g - greyRGB.g));
    const bVal = Math.round(greyRGB.b + t * (blueRGB.b - greyRGB.b));
    return `rgb(${rVal},${gVal},${bVal})`;
  };

  const batterKeys = [
    { key: "avg_exit_velo", label: "EV" },
    { key: "k_per", label: "K%" },
    { key: "bb_per", label: "BB%" },
    { key: "la_sweet_spot_per", label: "LA Sweet Spot %" },
    { key: "hard_hit_per", label: "Hard Hit %" },
    { key: "whiff_per", label: "Whiff %" },
    { key: "chase_per", label: "Chase %" },
  ];

  const pitcherKeys = [
    { key: "era", label: "ERA" },
    { key: "whip", label: "WHIP" },
    { key: "k_per", label: "K%" },
    { key: "bb_per", label: "BB%" },
    { key: "csw_per", label: "CSW%" },
    { key: "hard_hit_per", label: "Hard Hit %" },
    { key: "avg_exit_velo_allowed", label: "EV Allowed" },
  ];

  if (loading) return <div style={{ textAlign: "center", padding: 32 }}>Loading...</div>;
  if (error) return <div style={{ color: "#d32f2f", textAlign: "center", padding: 32 }}>{error}</div>;
  if (!stats) return <div style={{ textAlign: "center", padding: 32 }}>No data available</div>;

  const statKeys = role === "batter" ? batterKeys : pitcherKeys;

  return (
    <div style={{ display: "flex", width: "100%", maxWidth: 1200, margin: "40px auto", gap: 24 }}>
      <div style={boxStyle}>
        <div style={{ width: "100%", maxWidth: 400 }}>
          <h2 style={{ textAlign: "center", marginBottom: 24 }}>Advanced Stats</h2>
          {statKeys.map(({ key, label }) => {
            const rankKey = `${key}_rank`;
            let rank = typeof stats[rankKey] === "number" ? stats[rankKey] : 1;
            rank = Math.round(rank);

            const statValue =
              typeof stats[key] === "number"
                ? key.endsWith("per") || key === "k_per" || key === "bb_per"
                  ? `${(stats[key] * 100).toFixed(1)}%`
                  : stats[key].toFixed(2)
                : "0.0";

            return <StatBar key={key} statName={label} percentile={rank} color={getRankColor(rank)} statValue={statValue} />;
          })}

          {/* Infield slices for batters */}
          {role === "batter" && (
            <>
              <h2 style={{ textAlign: "center", margin: "32px 0 16px" }}>Infield Slices</h2>
              <InfieldSprayChart stats={stats} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
