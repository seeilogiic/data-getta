import Box from '@mui/material/Box';
import Link from '@/utils/Link';
import { useLocation } from 'react-router';
import { useState, useEffect } from 'react';

export default function TableTabs({ trackmanAbbreviation }: { trackmanAbbreviation: string }) {
  const currentURL = '/team/';
  const location = useLocation();

  const [batterUnderline, setBatterUnderline] = useState<'none' | 'hover' | 'always' | undefined>(
    'hover',
  );
  const [pitcherUnderline, setPitcherUnderline] = useState<'none' | 'hover' | 'always' | undefined>(
    'hover',
  );

  useEffect(() => {
    setBatterUnderline('hover');
    setPitcherUnderline('hover');

    if (location.pathname.includes('/batting')) {
      setBatterUnderline('always');
    } else if (location.pathname.includes('/pitching')) {
      setPitcherUnderline('always');
    }
  }, [location.pathname]);

  return (
    <Box
      sx={{
        display: 'flex',
        columnGap: 8,
        rowGap: 2,
        flexWrap: 'wrap',
      }}
    >
      <Link
        href={`${currentURL}${trackmanAbbreviation}/batting`}
        name="Batting"
        fontWeight={600}
        underline={batterUnderline}
      />
      <Link
        href={`${currentURL}${trackmanAbbreviation}/pitching`}
        name="Pitching"
        fontWeight={600}
        underline={pitcherUnderline}
      />
    </Box>
  );
}
