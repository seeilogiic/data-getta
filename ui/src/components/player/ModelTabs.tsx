import Box from '@mui/material/Box';
import Link from '@/utils/Link';
import { useLocation } from 'react-router';
import { useState, useEffect } from 'react';

export default function ModelTabs({
  team,
  player,
  role, // guaranteed to be 'batter' | 'pitcher' from PlayerPage
}: {
  team: string;
  player: string;
  role: 'batter' | 'pitcher';
}) {
  const baseURL = `/team/${team}/player/${player}`;
  const location = useLocation();
  const pathName = location.pathname;

  const [statsUnderline, setStatsUnderline] = useState<'none' | 'hover' | 'always'>('hover');
  const [heatMapUnderline, setHeatMapUnderline] = useState<'none' | 'hover' | 'always'>('hover');
  const [percentilesUnderline, setPercentilesUnderline] = useState<'none' | 'hover' | 'always'>('hover');

  // Set underline based on current path
  useEffect(() => {
    setStatsUnderline('hover');
    setHeatMapUnderline('hover');
    setPercentilesUnderline('hover');

    if (pathName.includes('/stats')) setStatsUnderline('always');
    else if (pathName.includes('/heat-map')) setHeatMapUnderline('always');
    else if (pathName.includes('/percentiles')) setPercentilesUnderline('always');
  }, [pathName]);

  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
      <Link
        href={`${baseURL}/stats/2025?role=${role}`}
        name="Stats"
        fontWeight={600}
        underline={statsUnderline}
      />
      <Link
        href={`${baseURL}/heat-map/2025?role=${role}`}
        name="Heatmaps"
        fontWeight={600}
        underline={heatMapUnderline}
      />
      <Link
        href={`${baseURL}/percentiles/2025?role=${role}`}
        name="Percentiles"
        fontWeight={600}
        underline={percentilesUnderline}
      />
    </Box>
  );
}
