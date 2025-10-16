import Box from '@mui/material/Box';
import { useState, useEffect } from 'react';
import { useParams, useSearchParams } from 'react-router';
import { CircularProgress, Typography } from '@mui/material';
import { supabase } from '@/utils/supabase/client';
import BatterHeatMap from '@/components/player/HeatMap/BatterHeatMap';
import PitcherHeatMap from '@/components/player/HeatMap/PitcherHeatMap';
import { BatterPitchBinsTable, PitcherPitchBinsTable } from '@/types/schemas';

export default function HeatMapTab() {
  const { trackmanAbbreviation, playerName, year } = useParams<{
    trackmanAbbreviation: string;
    playerName: string;
    year: string;
  }>();

  const [searchParams] = useSearchParams();
  const roleParam = searchParams.get('role');
  const role: 'batter' | 'pitcher' | null =
    roleParam === 'batter' || roleParam === 'pitcher' ? roleParam : null;

  const [batterBins, setBatterBins] = useState<BatterPitchBinsTable[]>([]);
  const [pitcherBins, setPitcherBins] = useState<PitcherPitchBinsTable[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const decodedTrackmanAbbreviation = trackmanAbbreviation
    ? decodeURIComponent(trackmanAbbreviation)
    : '';
  const decodedPlayerName = playerName
    ? decodeURIComponent(playerName).split('_').join(', ')
    : '';

  useEffect(() => {
    async function fetchData() {
      if (!role) {
        setError('Please specify a valid role in the URL query parameter: ?role=batter or ?role=pitcher');
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        if (role === 'batter') {
          const { data, error } = await supabase
            .from('BatterPitchBins')
            .select('*')
            .eq('Batter', decodedPlayerName)
            .eq('Year', Number(year))
            .eq('BatterTeam', decodedTrackmanAbbreviation)
            .overrideTypes<BatterPitchBinsTable[], { merge: false }>();

          if (error) throw error;
          setBatterBins(data || []);
        } else if (role === 'pitcher') {
          const { data, error } = await supabase
            .from('PitcherPitchBins')
            .select('*')
            .eq('Pitcher', decodedPlayerName)
            .eq('Year', Number(year))
            .eq('PitcherTeam', decodedTrackmanAbbreviation)
            .overrideTypes<PitcherPitchBinsTable[], { merge: false }>();

          if (error) throw error;
          setPitcherBins(data || []);
        }
      } catch (e: any) {
        console.error('Error fetching heat-map data:', e);
        setError(e.message || 'Failed to load heat-map data');
      } finally {
        setLoading(false);
      }
    }

    if (decodedTrackmanAbbreviation && decodedPlayerName && year) {
      fetchData();
    }
  }, [decodedTrackmanAbbreviation, decodedPlayerName, year, role]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: '4rem' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography variant="h6" color="#d32f2f" sx={{ py: '2rem', textAlign: 'center' }}>
        <strong>Error:</strong> {error}
      </Typography>
    );
  }

  if (role === 'batter' && !batterBins.length) {
    return (
      <Typography variant="body1" color="text.secondary" sx={{ py: '2rem', textAlign: 'center' }}>
        No batter heat-map data available.
      </Typography>
    );
  }

  if (role === 'pitcher' && !pitcherBins.length) {
    return (
      <Typography variant="body1" color="text.secondary" sx={{ py: '2rem', textAlign: 'center' }}>
        No pitcher heat-map data available.
      </Typography>
    );
  }

  return (
    <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', py: '2rem' }}>
      {role === 'batter' && <BatterHeatMap data={batterBins} />}
      {role === 'pitcher' && <PitcherHeatMap data={pitcherBins} />}
    </Box>
  );
}
