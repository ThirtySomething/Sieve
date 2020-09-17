//******************************************************************************
// Copyright 2020 ThirtySomething
//******************************************************************************
// This file is part of Sieve.
//
// Sieve is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Sieve is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
// more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Sieve. If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#ifndef SIEVEUI_H
#define SIEVEUI_H

#include <QMainWindow>
#include "csieve.h"
#include <future>

QT_BEGIN_NAMESPACE
namespace Ui { class SieveUI; }
QT_END_NAMESPACE

class SieveUI : public QMainWindow
{
	Q_OBJECT

public:
	SieveUI(QWidget* parent = nullptr);
	~SieveUI();

public slots:
	void setPrime(long long prime);

signals:
	void primeChanged(long long newPrime);

private slots:
	void on_actionQuit_triggered();

	void on_actionLoad_triggered();

	void on_actionSave_triggered();
	
	void on_actionNew_triggered();

	void on_actionAbout_Sieve_triggered();

	void on_btnStart_clicked();

	void on_btnStop_clicked();

private:
	Ui::SieveUI* ui;
	std::unique_ptr<net::derpaul::sieve::CSieve> m_sieve;
	std::future<void> m_processSieve;
};
#endif // SIEVEUI_H
