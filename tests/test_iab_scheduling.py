import threading
from concurrent.futures import ThreadPoolExecutor
from scripts import download_iab_subset as d


def test_real_and_ai_both_start_and_fast_ai_does_not_fill_slots_with_real(monkeypatch,tmp_path):
    release=threading.Event();guard=threading.Lock();started=[];ai_done=[]
    timer=threading.Timer(3,release.set);timer.start()
    real=[('real',str(i)) for i in range(4)];ai=[('model',str(i)) for i in range(5)]
    selections={key:[] for key in real+ai};records={key:key for key in selections}
    def acquire(record,*args):
        with guard:started.append(record)
        if record[0]=='real':
            assert release.wait(4)
        else:
            with guard:
                ai_done.append(record)
                if len(ai_done)==len(ai):
                    assert sum(k[0]=='real' for k in started)==2
                    release.set()
        return record
    monkeypatch.setattr(d,'acquire_cell',acquire)
    try:
        with ThreadPoolExecutor(4) as pool:
            results=[future.result() for key,future in d.acquisition_jobs(pool,records,selections,None,tmp_path,None,4)]
        assert set(results)==set(selections) and len(results)==len(selections)
        assert len(ai_done)==5
        assert sum(k[0]=='real' for k in started[:4])==2
    finally:release.set();timer.cancel()
